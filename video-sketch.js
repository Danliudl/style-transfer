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
let styleImg;
let outputImgData;
let outputImg;
let modelNum = 0;
let currentModel = "fuchun";
let webcam = false;
let modelReady = false;
let localVideo;
let videoUploader;
let video;
let isLoading = true;
let styleUploader;
let mobileStyleNet, separableTransformNet;
let isArbitrary = false;

function setup() {
  noCanvas();
  frameRate(20);
  noLoop();
  styleImg = select("#styleImage");

  // load models
  modelNames.forEach(n => {
    nets[n] = new ml5.TransformNet("models/" + n + "/", modelLoaded);
  });

  // video uploader

  videoUploader = select("#videoUploader").elt;
  videoUploader.addEventListener("change", gotNewInputVideo);

  // output img container
  outputImgContainer = createImg("images/loading.gif", "image");
  outputImgContainer.parent("output-img-container");

  // style uploader
  styleUploader = select("#uploadStyle").elt;
  styleUploader.addEventListener("change", gotNewStyle);

  // load
  Promise.all([
    loadMobileNetStyleModel(),
    loadSeparableTransformerModel()
  ]).then(([styleNet, transformNet]) => {
    console.log("Loaded styleNet");
    mobileStyleNet = styleNet;
    separableTransformNet = transformNet;
    // var constraints = {
    //   audio: false,
    //   video: { width: 200, height: 250 }
    // };
    // navigator.mediaDevices
    //   .getUserMedia(constraints)
    //   .then(function(mediaStream) {
    //     console.log("getUserMedia:", mediaStream);
    //     var video = document.getElementById("video");
    //     video.srcObject = mediaStream;
    //     video.onloadedmetadata = function(e) {
    //       video.play();
    //     };
    //   });
  });
}

// A function to be called when the model has been loaded
function modelLoaded() {
  modelNum++;
  if (modelNum >= modelNames.length) {
    modelReady = true;
  }
}

function predictVideo(modelName) {
  isLoading = true;
  if (!modelReady) return;
  if (webcam && video) {
    outputImgData = nets[modelName].predict(video.elt);
  } else if (localVideo) {
    outputImgData = nets[modelName].predict(localVideo.elt);
  }
  outputImg = ml5.array3DToImage(outputImgData);
  outputImgContainer.elt.src = outputImg.src;
  isLoading = false;
}

function draw() {
  if (modelReady) {
    if (isArbitrary) {
      arbitrary().finally(() => {
        console.log("finish");
      });
    } else {
      predictVideo(currentModel);
    }
  }
}

function updateStyleImg(ele) {
  select(".style-chosen").removeClass("style-chosen");
  if (ele.src) {
    currentModel = ele.id;
    select("#" + currentModel).addClass("style-chosen");
    isArbitrary = false;
    select("#stylized").hide();
    select("#output-img-container").show();
    if (video || localVideo) {
      predictVideo(currentModel);
      loop();
    }
  }
}

function uploadVideo() {
  videoUploader.click();
}

function gotNewInputVideo() {
  if (videoUploader.files && videoUploader.files[0]) {
    useVideo(window.URL.createObjectURL(videoUploader.files[0]));
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

function useVideo(url) {
  noLoop();
  select("#stylized").hide();
  if (!localVideo) {
    localVideo = createVideo([url], vidLoad);
    localVideo.size(320, 320);
    localVideo.parent("input-source");
  }
  webcam = false;
  function vidLoad() {
    localVideo.loop();
    localVideo.volume(0);
  }
  if (video) {
    video.remove();
  }
  //outputImgContainer.addClass("reverse-img");
}

function useWebcam() {
  noLoop();
  select("#stylized").hide();
  if (!video) {
    // webcam video
    video = createCapture(VIDEO);
    video.size(320, 320);
    video.parent("input-source");
  }
  webcam = true;
  if (localVideo) {
    localVideo.remove();
  }
  //select("#input-source").addClass("reverse-img");
  //outputImgContainer.addClass("reverse-img");
}

// function deactiveWebcam() {
//   noLoop();
//   // select("#input-source").removeClass("reverse-img");
//   // outputImgContainer.removeClass("reverse-img");
//   webcam = false;
//   if (video) {
//     video..remove();
//   }
// }
function onPredictVideoClick() {
  select("#stylized").show();
  if (isArbitrary) {
    arbitrary().finally(() => {
      console.log("finish");
    });
  } else {
    predictVideo(currentModel);
  }
  loop();
}

function saveShot() {
  noLoop();
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
  link.setAttribute("download", "output");
  link.click();
}
function uploadStyleImg() {
  noLoop();
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
          .fromPixels(document.querySelector("video"))
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
