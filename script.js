

/*
    references

    https://codelabs.developers.google.com/your-first-webgpu-app#7

    http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
*/

const GRID_SIZE = 64;
const UPDATE_INTERVAL = 50; // ms
let step = 0; // number of simulation steps

const CS_WORKGROUP_SIZE = 8; // for compute shaders

/* WebGPU variables ----------------------------------- */

const vertices = new Float32Array([
    //   X,    Y,
      -1.0, -1.0, // Triangle 1 (Blue)
       1.0, -1.0,
       1.0,  1.0,
    
      -1.0, -1.0, // Triangle 2 (Red)
       1.0,  1.0,
      -1.0,  1.0,
]);

var canvas = document.getElementById("surface");        // device is our interface with the GPU
var canvasFormat;

var device;
var context;
var encoder;
var adapter;

var vertexBuffer;
var vertexBufferLayout;
var cellShaderModule;
var cellPipeline;

const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
var uniformBuffer;
var bindGroup;

const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
var cellStateStorage;

var simulationShaderModule;

// -------------------------------------------------------

async function Init(){

    if(!navigator.gpu){
        throw new Error("WebGPU not supported on this browser");            
    }
    
    adapter = await navigator.gpu.requestAdapter();
    if(!adapter){
        throw new Error("No appropriate GPUAdapter found");                 
    }

    device = await adapter.requestDevice(); // device is our interface with the GPU

    canvas = document.getElementById("surface");
    context = canvas.getContext("webgpu");

    canvasFormat = navigator.gpu.getPreferredCanvasFormat();      // determine the preferred texture format for the canvas
    context.configure({
        device: device,
        format: canvasFormat,
    });


    encoder = device.createCommandEncoder();

}

function DefineBuffers(){

    vertexBuffer = device.createBuffer({
        label: "Cell vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });   

    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

    vertexBufferLayout = {
        arrayStride: 8,
        attributes: [{
          format: "float32x2",
          offset: 0,
          shaderLocation: 0, // Position, see vertex shader
        }],
    };

    cellShaderModule = device.createShaderModule({
        label: "Cell shader",
        code: `

            struct VertexInput {
                @location(0) pos: vec2f,
                @builtin(instance_index) instance : u32,
            };

            struct VertexOutput{
                @builtin(position) pos : vec4f,
                @location(0) cell : vec2f,
            };


            @group(0) @binding(0) var<uniform> grid: vec2f;             // size of the grid
            @group(0) @binding(1) var<storage> cellState: array<u32>;
            // note both uniforms use the same group (with different bindings)

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                
            
                let i = f32(input.instance);
                
                        // compute the cell position (x, y) from the cell index
                let cell = vec2f(i % grid.x, floor(i / grid.x));

                let cellOffset = cell / grid * 2;

                let state = f32(cellState[input.instance]); 

                let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

                var output: VertexOutput;
                output.pos = vec4f(gridPos, 0,1);
                output.cell = cell;

                return output;
            }

            struct FragInput {
                @location(0) cell: vec2f,
            };


            @fragment
            fn fragmentMain(input : FragInput) -> @location(0) vec4f {
                
                let c = input.cell / grid;
                return vec4f(c, 1 - c.x, 1);
            }
        `
    });

    // create a render pipeline from our shaders
    cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: "auto",

        vertex: {
          module: cellShaderModule,
          entryPoint: "vertexMain",
          buffers: [vertexBufferLayout]
        },

        fragment: {
          module: cellShaderModule,
          entryPoint: "fragmentMain",
          targets: [{
            format: canvasFormat
          }]
        }
    });

    
    // Create a uniform buffer that describes the grid.

    uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    cellStateStorage = [
        
        device.createBuffer({
        // two buffers to switch between (allows writing to opposite then switching)
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,

        }),
        device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,

        })
    ];

    // Mark every third cell of the grid as active.
    for (let i = 0; i < cellStateArray.length; i += 5) {
        cellStateArray[i] = 1;
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    for (let i = 0; i < cellStateArray.length; i += 8) {
        cellStateArray[i] = 1;
    }
    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);



    // Create the compute shader that will process the simulation.
    simulationShaderModule = device.createShaderModule({
        label: "Compute Shader",
        code: `
        

        @group(0) @binding(0) var<uniform> grid: vec2f; 

        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;                     // read only                   
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;        // read and write (we write here)
                                                                                        // these buffers switch places the next iteration

        // given a 
        fn cellIndex(cell: vec2u) -> u32 {
            return cell.y * u32(grid.x) + cell.x;
        }


        @compute
                    // specifies we will work in 8x8x1 groups (z defaults to 1 because is not specified)
        @workgroup_size(${CS_WORKGROUP_SIZE}, ${CS_WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell : vec3u) {
            
            // global_invocation_id is essentially the cell we are operating on
            // (0,0,0) (x, y, z)

        }`
    });



    // creating a bindGroup to "bind" the uniform to our shader
    bindGroup = 
    
    [
        device.createBindGroup({
            label: "Cell renderer bind group",
            layout: cellPipeline.getBindGroupLayout(0),
            entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[0] }
            }
            
            ],
        }),

        device.createBindGroup({
            label: "Cell renderer bind group B",
            layout: cellPipeline.getBindGroupLayout(0),
            entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: cellStateStorage[1] }
            }
            
            ],
        })

    ];
}

function Draw(){

    step++;
    
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),  // canvas texture is given as the view property
            loadOp: "clear",                                 // indicates we want the texture cleared at the start of the render pass
            storeOp: "store",                                // indicates we want to the results of any drawing to be saved into the texture
            clearValue: [0.14, 0.08, 0.2, 1],                // background colour
        }]
    })

    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);

    pass.setBindGroup(0, bindGroup[step % 2]); 

                                // specify we want an instance for each grid cell
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices

    pass.end();

    device.queue.submit([encoder.finish()]); // communicate recorded commands to GPU
}

async function Run(){

    await Init();
    DefineBuffers();

    setInterval(Draw, UPDATE_INTERVAL);
}

Run();
