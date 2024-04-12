importScripts(
  "./lib/tf.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.17.0/dist/tf-backend-wasm.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.17.0/dist/tf-backend-webgpu.min.js",
);

let device = "wasm";
let model = null;

if (navigator.gpu) device = "webgpu";
tf.setBackend(device).then(async () => {
  model = await tf.loadGraphModel("./model/model.json");
  console.log("model", model);
  let threadsCount = 0;
  if (device === "wasm") {
    try {
      threadsCount = tf.wasm.getThreadsCount();
    } catch (e) {
      console.log("Error", e);
    }
  }
  postMessage({ type: "modelLoaded", threadsCount, device });
});

async function run_model(input) {
  if (!model) {
    model = await model;
  }
  tf_img = tf.browser.fromPixels(input);
  input = tf_img.div(255.0).expandDims().toFloat();
  const outputs = await model.predict(input);
  return outputs.dataSync();
}

onmessage = async (event) => {
  const { input, startTime } = event.data;
  const output = await run_model(input);
  postMessage({ type: "modelResult", result: output, startTime });
};
