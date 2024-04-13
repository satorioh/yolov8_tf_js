importScripts(
  "./lib/tf.min.js",
  "./lib/tf-backend-wasm.min.js",
  "./lib/tf-backend-webgpu.min.js",
);

let device = "wasm";
let model = null;

async function init() {
  if (navigator.gpu && (await navigator.gpu.requestAdapter())) {
    device = "webgpu";
  } else {
    tf.wasm.setWasmPaths(
      "https://regulussig.s3.ap-southeast-1.amazonaws.com/tfjs/wasm/",
    );
  }
  load_model();
}

init();

async function load_model() {
  await tf.setBackend(device);
  model = await tf.loadGraphModel(
    "https://regulussig.s3.ap-southeast-1.amazonaws.com/tfjs/model/model.json",
  );
  console.log("model loaded", model);
  let threadsCount = 0;
  if (device === "wasm") {
    try {
      threadsCount = tf.wasm.getThreadsCount();
    } catch (e) {
      console.log("getThreadsCount Error", e);
    }
  }
  postMessage({ type: "modelLoaded", threadsCount, device });
}

async function run_model(input) {
  if (!model) {
    model = await model;
  }
  tf_img = tf.browser.fromPixels(input);
  input = tf_img.div(255.0).expandDims().toFloat();
  const outputs = await model.predict(input);
  return outputs.data();
}

onmessage = async (event) => {
  const { input, startTime } = event.data;
  const output = await run_model(input);
  postMessage({ type: "modelResult", result: output, startTime });
};
