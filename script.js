

/*
    references

    https://codelabs.developers.google.com/your-first-webgpu-app#7

    http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
*/

const GRID_SIZE = 64;
const UPDATE_INTERVAL = 5; // ms
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
var adapter;

var vertexBuffer;
var vertexBufferLayout;
var cellShaderModule;
var cellPipeline;

const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
var uniformBuffer;
var bindGroups;

var pipelineLayout;
var bindGroupLayout;

const densityStateArray = new Float32Array(GRID_SIZE * GRID_SIZE);
const velocityStateArray = new Float32Array(GRID_SIZE * GRID_SIZE * 2); // * 2 to allow (x, y) values (two elements make one vector)

var densityStateStorage; // [0] == front buffer, [1] == back buffer (switch between each iteration)
var velocityStateStorage;

var simulationShaderModule;
var simulationPipeline;

var mouseDown = false;
var mouseCoordX = 0;
var mouseCoordY = 0;

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
                @location(1) @interpolate(flat) instance : u32,
            };


            @group(0) @binding(0) var<uniform> grid: vec2f;             // size of the grid
            @group(0) @binding(1) var<storage> densityIn: array<f32>;
            // note both uniforms use the same group (with different bindings)

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                
                let i = f32(input.instance);
                
                        // compute the cell position (x, y) from the cell index
                let cell = vec2f(i % grid.x, floor(i / grid.x));

                let cellOffset = cell / grid * 2;


                let gridPos = (input.pos + 1) / grid - 1 + cellOffset;

                var output: VertexOutput;
                output.pos = vec4f(gridPos, 0,1);
                output.cell = cell;
                output.instance = input.instance;

                return output;
            }

            struct FragInput {
                @location(0) cell: vec2f,
                @location(1) @interpolate(flat) instance : u32,
            };


            @fragment
            fn fragmentMain(input : FragInput) -> @location(0) vec4f {
                
                let density = f32(densityIn[input.instance]); 

                return vec4f(density, density, density, 1);
            }
        `
    });

    // Create a uniform buffer that describes the grid.

    uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    densityStateStorage = [
        
        device.createBuffer({
        // two buffers to switch between (allows writing to opposite then switching)
            label: "Density State A",
            size: densityStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,

        }),
        device.createBuffer({
            label: "Density State B",
            size: densityStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
    ];
    
    velocityStateStorage = [
        
        device.createBuffer({
        // two buffers to switch between (allows writing to opposite then switching)
            label: "Velocity State A",
            size: velocityStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,

        }),
        device.createBuffer({
            label: "Velocity State B",
            size: velocityStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
    ];


    for (let i = 0; i < densityStateArray.length; i ++) {
        densityStateArray[i] = 0;
    }
    densityStateArray[100] = 1;
    densityStateArray[25] = 1;
    device.queue.writeBuffer(densityStateStorage[0], 0, densityStateArray);
    device.queue.writeBuffer(densityStateStorage[1], 0, densityStateArray);

    for (let i = 0; i < velocityStateArray.length; i ++) {
        velocityStateArray[i] = 0;
    }
    device.queue.writeBuffer(velocityStateStorage[0], 0, velocityStateArray);
    device.queue.writeBuffer(velocityStateStorage[1], 0, velocityStateArray);


    // Create the compute shader that will process the simulation.
    simulationShaderModule = device.createShaderModule({
        label: "Compute Shader",
        code: `
        

        @group(0) @binding(0) var<uniform> grid: vec2f; 

        @group(0) @binding(1) var<storage> densityIn: array<f32>;                     // read only                   
        @group(0) @binding(2) var<storage, read_write> densityOut: array<f32>;        // read and write (we write here)

        @group(0) @binding(3) var<storage> velocityIn: array<f32>;                               
        @group(0) @binding(4) var<storage, read_write> velocityOut: array<f32>;        

                                                                                        // these buffers switch places the next iteration

        // mapping an x,y coordinate to a index (for the storage arrays ^)
        fn cellIndex(cell: vec2i) -> i32 {
            return cell.y * i32(grid.x) + cell.x;
        }


        @compute
                    // specifies we will work in 8x8x1 groups (z defaults to 1 because is not specified)
        @workgroup_size(${CS_WORKGROUP_SIZE}, ${CS_WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell : vec3u) {
            
            // global_invocation_id is essentially the cell we are operating on
            // (0,0,0) (x, y, z)

            let coord = vec2i(i32(cell.x), i32(cell.y));

            let index = cellIndex(coord);

            var left: f32 = 0.0;
            var right: f32 = 0.0;
            var top: f32 = 0.0;
            var bottom: f32 = 0.0;
            var sampleCount = 0;

            // Sample neighboring cells, ensuring we're within bounds
            if (coord.x > 0) {
                left = densityIn[cellIndex(coord + vec2i(-1, 0))];
                sampleCount++;
            }
            if (coord.x < i32(grid.x) - 1) {
                right = densityIn[cellIndex(coord + vec2i(1, 0))];
                sampleCount++;
            }
            if (coord.y > 0) {
                top = densityIn[cellIndex(coord + vec2i(0, -1))];
                sampleCount++;
            }
            if (coord.y < i32(grid.y) - 1) {
                bottom = densityIn[cellIndex(coord + vec2i(0, 1))];
                sampleCount++;
            }            

            let avgDensity = (left + right + top + bottom) / 4.0;

            densityOut[index] = avgDensity;

        }`
    });



    // Create the bind group layout and pipeline layout.
    bindGroupLayout = device.createBindGroupLayout({
        label: "Cell Bind Group Layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
            buffer: {} // Grid uniform buffer
        }, 
        {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
            buffer: { type: "read-only-storage"} // density state input buffer
        }, 
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage"} // density state output buffer
        },
        {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage"} // velocity state input buffer
        }, 
        {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage"} // velocity state output buffer
        }
    ]
    });

    pipelineLayout = device.createPipelineLayout({
        label: "Cell Pipeline Layout",
        bindGroupLayouts: [ bindGroupLayout ],
      });

    // create a render pipeline from our shaders
    cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: pipelineLayout,

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

    simulationPipeline = device.createComputePipeline({
        label: "Simulation pipeline",
        layout: pipelineLayout,
        compute: {
          module: simulationShaderModule,
          entryPoint: "computeMain",
        }
    });

    // creating a bindGroup to "bind" the uniform to our shader
    bindGroups = 
    
    [
        device.createBindGroup({
            label: "Cell renderer bind group",
            layout: bindGroupLayout,
            
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: densityStateStorage[0] }
            },
            {
                binding: 2, 
                resource: { buffer: densityStateStorage[1] }
            },
            {
                binding: 3,
                resource: { buffer: velocityStateStorage[0] }
            },
            {
                binding: 4, 
                resource: { buffer: velocityStateStorage[1] }
            },
            ],
        }), 

        // two bind groups are created to switch between iteration (we read from one, write to the other...)

        device.createBindGroup({
            label: "Cell renderer bind group B",
            layout: bindGroupLayout,
            
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            {
                binding: 1,
                resource: { buffer: densityStateStorage[1] }
            },
            {
                binding: 2,
                resource: { buffer: densityStateStorage[0] }
            },
            
            {
                binding: 3,
                resource: { buffer: velocityStateStorage[1] }
            },
            {
                binding: 4, 
                resource: { buffer: velocityStateStorage[0] }
            },
            ],
        })

    ];
}

function ProcessFrame(){

    Process_InsertDensity();

    const encoder = device.createCommandEncoder();

    ProcessFrame_Compute(encoder);
    ProcessFrame_Draw(encoder);

    device.queue.submit([encoder.finish()]); // communicate recorded commands to GPU
    
}

function CoordinateToIndex(x, y) {

    y = GRID_SIZE - y;
    return y * GRID_SIZE + x;
}

function Process_InsertDensity(){
    
    if(!mouseDown){
        return;
    }

    let index = CoordinateToIndex(mouseCoordX, mouseCoordY);

    //densityStateArray[index] = 1;

    device.queue.writeBuffer(
        densityStateStorage[(step - 1) % 2],
        index * 4,  // Offset for the specific index in the buffer
        new Float32Array([1.0]), // Write only the density at this index
    );
    device.queue.writeBuffer(
        densityStateStorage[(step) % 2],
        index * 4,  // Offset for the specific index in the buffer
        new Float32Array([1.0]), // Write only the density at this index
    );
    /*
    // write new density to buffer
    device.queue.writeBuffer(
        densityStateStorage[(step - 1) % 2], 
        index * 4,
        densityStateArray,
        index,
        2,
    );

    device.queue.writeBuffer(
        densityStateStorage[(step) % 2], 
        index * 4,
        densityStateArray,
        index,
        2,
    );
    */
}

function ProcessFrame_Compute(encoder){
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    /*
        Something very important to note here is that the number you pass into dispatchWorkgroups() is not the number of invocations! Instead, 
        it's the number of workgroups to execute, as defined by the @workgroup_size in your shader.

        If you want the shader to execute 32x32 times in order to cover your entire grid, and your workgroup size is 8x8, 
        you need to dispatch 4x4 workgroups (4 * 8 = 32). That's why you divide the grid size by the workgroup size and pass 
        that value into dispatchWorkgroups().
    */
    const workgroupCount = Math.ceil(GRID_SIZE / CS_WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();
}

function ProcessFrame_Draw(encoder){

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

    pass.setBindGroup(0, bindGroups[step % 2]); 

                                // specify we want an instance for each grid cell
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices

    pass.end();

}

function UpdateMousePosition(event){
    // store mouse coordinate

    let x = event.offsetX;
    let y = event.offsetY;

    // handle bounds

    if(x < 0 || x >= 512){
        return;
    }
    if(y < 0 || y >= 512){
        return;
    } 
                                        // canvas width
    mouseCoordX = Math.floor(x * (GRID_SIZE / 512));
    mouseCoordY = Math.floor(y * (GRID_SIZE / 512));
}

async function Run(){

    await Init();
    DefineBuffers();

    addEventListener("mousedown", (event) => {
        mouseDown = true;
        UpdateMousePosition(event);
    });

    addEventListener("mouseup", (event) => {
        mouseDown = false;
    });

    addEventListener("mousemove", (event) => {

        UpdateMousePosition(event);
    });

    setInterval(ProcessFrame, UPDATE_INTERVAL);
}

Run();
