importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.17.0/dist/tf-backend-wasm.min.js",
);

let model = null;
tf.setBackend("wasm").then(async () => {
  model = await tf.loadGraphModel("./model/model.json");
  console.log("model", model);
  postMessage({ type: "modelLoaded" });
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
