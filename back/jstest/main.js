
const http = require("http"), url = require("url");

const tf = require('@tensorflow/tfjs');
const posenet = require('@tensorflow-models/posenet');

const {createCanvas, Image} = require('canvas');

require('@tensorflow/tfjs-node');

global.XMLHttpRequest = require("xhr2");
global.fetch = require('node-fetch');

let load_model = (async () => {return await posenet.load().catch((error) => { console.log(error); })});

let load_model_fixed = (async () => {
    return await tf.loadFrozenModel(
        'http://127.0.0.1:1997/model/' + String(Math.random()) + '/file.pb',
        'http://127.0.0.1:1997/model_weights/' + String(Math.random()) + '/file.json',
        {mode: "cors", cache: "no-cache"});
});

const real_size = [23.40, 17.84];

load_model_fixed().then((fixed_model) => {

    load_model().then((model) => {

        http.createServer(function (req, res) {

            const img = new Image();
            img.src = '/home/sebastien/Project/conseil/CostConsulting/smartscan/back/data/test.png';
            img.onerror = () => {};

            const max = img.width > img.height ? img.width : img.height;
            const canvas = createCanvas(max, max);
            const ctx = canvas.getContext('2d');

            ctx.drawImage(img, 0, 0);

            const img_tensor = tf.fromPixels(canvas);

            let run = async () => { return await model.estimateSinglePose(img_tensor, 1, false, 32); };
            let run_fixed = async () => { return await fixed_model.executeAsync(tf.expandDims(img_tensor, 0)); };

            run().catch((error) => {console.log(error);}).then((pos) => {

                run_fixed().catch((error) => {console.log(error); }).then((fixed_results) => {

                    let box_real_coordinates = undefined;

                    const number_of_box = fixed_results[1].dataSync();

                    const vector_length = (v1, v2) => {return Math.sqrt(Math.pow(v1['x'] - v2['x'], 2) + Math.pow(v1['y'] - v2['y'], 2));};

                    if (number_of_box > 0) {

                        const box_raw_coordinate = fixed_results[0].dataSync();
                        box_real_coordinates = [
                            box_raw_coordinate[1] * max,
                            box_raw_coordinate[0] * max,
                            (box_raw_coordinate[3] - box_raw_coordinate[1]) * max,
                            (box_raw_coordinate[2] - box_raw_coordinate[0]) * max,
                        ];

                        const box_size = [box_real_coordinates[2], box_real_coordinates[3]];
                        const adjusted_size = [real_size[0], real_size[1]];
                        const adjusted = (box_size[0] / adjusted_size[0] + box_size[1] / adjusted_size[1]) / 2;

                        let body_part = {};

                        for (const x in pos['keypoints']) {
                            body_part[pos['keypoints'][x]['part']] = {
                                'x': pos['keypoints'][x]['position']['x'],
                                'y': pos['keypoints'][x]['position']['y'],
                            };
                        }

                        let lengths = {
                            'right_arm': vector_length(body_part['rightShoulder'], body_part['rightElbow']),
                            'right_forearm': vector_length(body_part['rightElbow'], body_part['rightWrist']),

                            'left_arm': vector_length(body_part['leftShoulder'], body_part['leftElbow']),
                            'left_forearm': vector_length(body_part['leftElbow'], body_part['leftWrist']),

                            'right_thigh': vector_length(body_part['rightHip'], body_part['rightKnee']),
                            'right_leg': vector_length(body_part['rightKnee'], body_part['rightAnkle']),

                            'left_thigh': vector_length(body_part['leftHip'], body_part['leftKnee']),
                            'left_leg': vector_length(body_part['leftKnee'], body_part['leftAnkle']),

                            'right_trunk': vector_length(body_part['rightShoulder'], body_part['rightHip']),
                            'left_trunk': vector_length(body_part['leftShoulder'], body_part['leftHip']),
                        };

                        for (const i in lengths) lengths[i] = lengths[i] / adjusted;

                        const unique_lenght = {
                            'arms': (lengths['right_arm'] + lengths['left_arm']) / 2,
                            'forearms': (lengths['right_forearm'] + lengths['left_forearm']) / 2,
                            'thighs': (lengths['right_thigh'] + lengths['left_thigh']) / 2,
                            'legs': (lengths['right_leg'] + lengths['left_leg']) / 2,
                            'trunk': (lengths['left_trunk'] + lengths['right_trunk']) / 2
                        };

                        console.log(lengths);

                        console.log(unique_lenght);

                    }

                    const canvas_r = createCanvas(img.width, img.height);
                    const ctx_r = canvas_r.getContext('2d');

                    ctx_r.drawImage(img, 0, 0);
                    ctx_r.fillStyle = "#FF0000";

                    for (const x in pos['keypoints']) {
                        const position = {
                            'x': pos['keypoints'][x]['position']['x'],
                            'y': pos['keypoints'][x]['position']['y'],
                        };
                        ctx_r.fillRect(position['x'], position['y'], 10, 10);
                    }
                    if (number_of_box > 0) {
                        ctx_r.beginPath();
                        ctx_r.strokeStyle = "red";
                        ctx_r.rect(box_real_coordinates[0], box_real_coordinates[1],
                            box_real_coordinates[2], box_real_coordinates[3]);
                        ctx_r.stroke();
                    }
                    res.writeHead(200, {'Content-Type': 'image/png'});
                    res.end(canvas_r.toBuffer('image/png'), 'binary');

                });

            });

        }).listen(8888);

    });

}).catch((error) => { console.log(error); });