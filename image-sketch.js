let nets = {};
let modelNames = [
  "fuchun",
  "la_muse",
  "mathura",
  // "matilde_perez",
  "matta",
  "rain_princess",
  "scream",
  "udnie",
  "wave",
  "wreck",
  "zhangdaqian"
];
let inputImg, styleImg;
let outputImgData;
let outputImg;
let modelNum = 0;
let currentModel = "fuchun";
let uploader;
let styleUploader;
let modelReady = false;
let isLoading = true;
let mobileStyleNet, separableTransformNet;
let isArbitrary = false;

function setup() {
  noCanvas();
  inputImg = select("#input-img").elt;
  styleImg = select("#styleImage");

  // load models
  modelNames.forEach(n => {
    nets[n] = new ml5.TransformNet("models/" + n + "/", modelLoaded);
  });

  // Image uploader
  uploader = select("#uploader").elt;
  uploader.addEventListener("change", gotNewInputImg);

  // style uploader
  styleUploader = select("#uploadStyle").elt;
  styleUploader.addEventListener("change", gotNewStyle);

  // output img container
  outputImgContainer = createImg("images/loading.gif", "image");
  outputImgContainer.parent("output-img-container");

  // load
  Promise.all([
    loadMobileNetStyleModel(),
    loadSeparableTransformerModel()
  ]).then(([styleNet, transformNet]) => {
    console.log("Loaded styleNet");
    mobileStyleNet = styleNet;
    separableTransformNet = transformNet;
  });
}

// A function to be called when the model has been loaded
function modelLoaded() {
  modelNum++;
  if (modelNum >= modelNames.length) {
    modelReady = true;
    predictImg(currentModel);
  }
}

function predictImg(modelName) {
  isLoading = true;
  if (!modelReady) return;
  if (inputImg) {
    outputImgData = nets[modelName].predict(inputImg);
  }
  outputImg = ml5.array3DToImage(outputImgData);
  outputImgContainer.elt.src = outputImg.src;
  isLoading = false;
}

function updateStyleImg(ele) {
  select(".style-chosen").removeClass("style-chosen");
  if (ele.src) {
    currentModel = ele.id;
    select("#" + currentModel).addClass("style-chosen");
    isArbitrary = false;
    select("#stylized").hide();
    select("#output-img-container").show();
    predictImg(currentModel);
  }
}

function updateInputImg(ele) {
  if (ele.src) inputImg.src = ele.src;
  if (isArbitrary) {
    arbitrary().finally(() => {
      console.log("finish");
    });
  } else {
    predictImg(currentModel);
  }
}

function uploadImg() {
  uploader.click();
}

function gotNewInputImg() {
  if (uploader.files && uploader.files[0]) {
    let newImgUrl = window.URL.createObjectURL(uploader.files[0]);
    inputImg.src = newImgUrl;
    inputImg.style.width = "300px";
    inputImg.style.height = "300px";
  }
}

function gotNewStyle() {
  if (styleUploader.files && styleUploader.files[0]) {
    styleImg.elt.src = window.URL.createObjectURL(styleUploader.files[0]);
    select(".style-chosen").removeClass("style-chosen");
    styleImg.addClass("style-chosen");
    isArbitrary = true;
    select("#stylized").show();
    select("#output-img-container").hide();
  }
}

function onPredictClick() {
  if (isArbitrary) {
    arbitrary().finally(() => {
      console.log("finish");
    });
  } else {
    predictImg(currentModel);
  }
}

function saveImg() {
  let link = document.createElement("a");
  if (isArbitrary) {
    link.setAttribute(
      "href",
      document
        .getElementById("stylized")
        .toDataURL("image/png")
        .replace("image/png", "image/octet-stream")
    );
  } else {
    link.setAttribute("href", outputImg.src);
  }

  link.setAttribute("download", "output.png");
  link.click();
}

function uploadStyleImg() {
  styleUploader.click();
}

async function loadMobileNetStyleModel() {
  if (!mobileStyleNet) {
    mobileStyleNet = await tf.loadGraphModel("saved_model_style_js/model.json");
  }

  return mobileStyleNet;
}
async function loadSeparableTransformerModel() {
  if (!separableTransformNet) {
    separableTransformNet = await tf.loadGraphModel(
      "saved_model_transformer_separable_js/model.json"
    );
  }

  return separableTransformNet;
}

async function arbitrary() {
  await tf.nextFrame();
  let bottleneck = await tf.tidy(() => {
    return mobileStyleNet.predict(
      tf.browser
        .fromPixels(document.getElementById("styleImage"))
        .toFloat()
        .div(tf.scalar(255))
        .expandDims()
    );
  });
  await tf.nextFrame();
  const stylized = await tf.tidy(() => {
    return separableTransformNet
      .predict([
        tf.browser
          .fromPixels(document.getElementById("input-img"))
          .toFloat()
          .div(tf.scalar(255))
          .expandDims(),
        bottleneck
      ])
      .squeeze();
  });
  await tf.browser.toPixels(stylized, document.getElementById("stylized"));
  bottleneck.dispose(); // Might wanna keep this around
  stylized.dispose();
}
