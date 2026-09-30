// ===============================
// Image Upload Prediction
// ===============================
function uploadImage() {
    let file = document.getElementById("upload").files[0];
    let formData = new FormData();
    formData.append("image", file);

    fetch("/predict_image", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("output").src =
            "data:image/jpeg;base64," + data.image;

        document.getElementById("classname").innerText =
            "Detected: " + data.class;
    })
    .catch(err => console.error("Upload error:", err));
}


// ===============================
// Live Webcam Prediction
// ===============================

let webcamStream = null;
let running = false;
let processing = false;

let frameSkip = 0;

const SKIP_FRAMES = 3;
const TARGET_WIDTH = 320;
const TARGET_HEIGHT = 240;
const JPEG_QUALITY = 0.4;


function startCamera() {

    let video = document.getElementById("webcam");

    navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "environment"
        },
        audio: false
    })
    .then(stream => {

        webcamStream = stream;

        video.srcObject = stream;

        running = true;
        processing = false;
        frameSkip = 0;

        processFrame();
    })
    .catch(err => {
        console.error("Camera error:", err);
    });
}


function stopCamera() {

    running = false;
    processing = false;

    if (webcamStream) {

        webcamStream.getTracks().forEach(track => {
            track.stop();
        });

        webcamStream = null;
    }
}


async function processFrame() {

    if (!running) {
        return;
    }

    let video = document.getElementById("webcam");

    if (
        video.readyState === video.HAVE_ENOUGH_DATA &&
        !processing
    ) {

        frameSkip++;

        if (frameSkip >= SKIP_FRAMES) {

            frameSkip = 0;

            processing = true;

            try {

                // -------------------------------
                // Create small frame
                // -------------------------------

                let canvas = document.createElement("canvas");

                canvas.width = TARGET_WIDTH;
                canvas.height = TARGET_HEIGHT;

                let ctx = canvas.getContext("2d", {
                    willReadFrequently: true
                });

                ctx.drawImage(
                    video,
                    0,
                    0,
                    TARGET_WIDTH,
                    TARGET_HEIGHT
                );


                // -------------------------------
                // Convert to JPEG
                // -------------------------------

                let data = canvas.toDataURL(
                    "image/jpeg",
                    JPEG_QUALITY
                );


                // -------------------------------
                // Send frame to Flask
                // -------------------------------

                let response = await fetch(
                    "/predict_frame",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify({
                            frame: data
                        })
                    }
                );


                let result = await response.json();


                // -------------------------------
                // Display result
                // -------------------------------

                if (result.frame) {

                    document.getElementById("output").src =
                        "data:image/jpeg;base64," + result.frame;
                }


                if (result.class) {

                    document.getElementById("classname").innerText =
                        "Detected: " + result.class;
                }


            } catch (err) {

                console.error(
                    "Prediction error:",
                    err
                );

            } finally {

                // VERY IMPORTANT
                // Allow next inference only
                // after current request finishes

                processing = false;
            }
        }
    }


    // Continue camera loop

    requestAnimationFrame(processFrame);
}
