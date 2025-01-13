






/* WebGPU variables ----------------------------------- */

const vertices = new Float32Array([
    //   X,    Y,
      -0.8, -0.8, // Triangle 1 (Blue)
       0.8, -0.8,
       0.8,  0.8,
    
      -0.8, -0.8, // Triangle 2 (Red)
       0.8,  0.8,
      -0.8,  0.8,
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

            @vertex
            fn vertexMain(@location(0) pos: vec2f) -> 
                @builtin(position) vec4f {
            
                return vec4f(pos, 0,1); 
                
            }

            @fragment
            fn fragmentMain() -> @location(0) vec4f {
                return vec4f(1, 0, 0, 1);
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

}

function Draw(){
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
    pass.draw(vertices.length / 2); // 6 vertices

    pass.end();

    device.queue.submit([encoder.finish()]); // communicate recorded commands to GPU
}


async function Run(){

    await Init();
    Draw();
}

Run();
