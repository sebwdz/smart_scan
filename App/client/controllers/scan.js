
const http = require("http"), url = require("url");

const tf = require('@tensorflow/tfjs');
const posenet = require('@tensorflow-models/posenet');
const {createCanvas} = require('canvas');

global.XMLHttpRequest = require("xhr2");
global.fetch = require('node-fetch');

/*

    CONSTANTS AND MODELS VARIABLES

*/

const real_size = [23.40, 17.84];

let posenet_model = undefined;
let fixed_model = undefined;

let found = false;
let count_objects = 0;
let stored_images = [];

let image_canvas = undefined;
let measures = {};

/*

    FUNCTION TO LOAD AND PROCESS MODELS

*/

const load_posnet_model = (async () => {
    return await posenet.load();
});

const load_model_fixed = (async () => {
    return await tf.loadFrozenModel('http://54.38.246.156:1997/model/' + String(Math.random()) + '/file.pb',
        'http://54.38.246.156:1997/model_weights/' + String(Math.random()) + '/file.json',
        {mode: "cors"});
});


const vector_length = (v1, v2) => {
    return Math.sqrt(Math.pow(v1['x'] - v2['x'], 2) + Math.pow(v1['y'] - v2['y'], 2));
};

const run_posnet = async (img_tensor) => {return await posenet_model.estimateSinglePose(img_tensor, 0.5, false, 32);};
const run_fixed = async (img_tensor) => {return await fixed_model.executeAsync(tf.expandDims(img_tensor, 0));};



export function loading_models(callback) {
    load_model_fixed().then((model) => {
        fixed_model = model;
        load_posnet_model().then((model) => {
            posenet_model = model;
            callback();
        });
    });

}

export function start() {

    image_canvas = null;
    stored_images = [];
    found = false;
    count_objects = 0;
    measures = {
        "box_width": [], "box_height": [],
        "arm": [], 'forearm': [],
        'thigh': [], 'leg': [],
        'trunk': [], 'shoulders': [], 'pelvis': []
    };

}

export function stop(processing_callback, final_callback) {

    function compute_image_size(image_it) {

        processing_callback((image_it / stored_images.length) * 100);

        if (image_it < stored_images.length) {
            extraction(stored_images[image_it], () => { compute_image_size(image_it + 1) });
        } else {
            finalize(final_callback);
        }
    }

    compute_image_size(0);
}

export function process_image(img, callback) {

    let max = 0;
    if (!image_canvas) {
        max = (img.width > img.height ? img.width : img.height) * 0.5;
        image_canvas = createCanvas(max, max);
    }

    image_canvas.getContext('2d').clearRect(0, 0, image_canvas.width, image_canvas.height);
    image_canvas.getContext('2d').drawImage(img, 0, 0, image_canvas.width, image_canvas.height);

    const img_tensor = tf.fromPixels(image_canvas);

    if (found) {
        if (callback) callback(true);
        stored_images.push(img);
    }
    else {
        run_fixed(img_tensor).then((fixed_results) => {

            if (fixed_results && fixed_results[1].dataSync() > 0) { count_objects++; stored_images.push(img); }
            if (count_objects > 2) found = true;
            if (callback) callback(found);

        }).catch(() => { if (callback) callback(found); });
    }
}

function extraction(image, callback) {

    const max = (image.width > image.height ? image.width : image.height) * 0.5;

    image_canvas.getContext('2d').clearRect(0, 0, max, max);
    image_canvas.getContext('2d').drawImage(image, 0, 0, image.width * 0.5, image.height * 0.5);
    const img_tensor = tf.fromPixels(image_canvas);

    measure_box();

    function measure_box() {

        run_fixed(img_tensor).then((fixed_results) => {

            if (fixed_results && fixed_results[1].dataSync() > 0) {
                const box_raw_coordinate = fixed_results[0].dataSync();
                measures['box_width'].push((box_raw_coordinate[3] - box_raw_coordinate[1]) * max);
                measures['box_height'].push((box_raw_coordinate[2] - box_raw_coordinate[0]) * max);
            }
            setTimeout(measure_person(), 5);

        }).catch(() => measure_person());

    }

    function measure_person() {

        const body_parts = {
            "right_arm": ['rightShoulder', 'rightElbow'],
            "left_arm": ['leftShoulder', 'leftElbow'],
            'right_forearm': ['rightElbow', 'rightWrist'],
            'left_forearm': ['leftElbow', 'leftWrist'],
            'right_thigh': ['rightHip', 'rightKnee'],
            'right_leg': ['rightKnee', 'rightAnkle'],
            'left_thigh': ['leftHip', 'leftKnee'],
            'left_leg': ['leftKnee', 'leftAnkle'],
            'right_trunk': ['rightShoulder', 'rightHip'],
            'left_trunk': ['leftShoulder', 'leftHip'],
            '_shoulders': ['leftShoulder', 'rightShoulder'],
            '_pelvis': ['leftHip', 'rightHip']
        };

        run_posnet(img_tensor).then((positions) => {

            let points = {};
            for (const x in positions['keypoints']) {
                if (positions['keypoints'][x]['score'] > 0.7) {
                    points[positions['keypoints'][x]['part']] = {
                        'x': positions['keypoints'][x]['position']['x'],
                        'y': positions['keypoints'][x]['position']['y'],
                    };
                }
            }
            for (const body_part in body_parts) {
                const point_1 = body_parts[body_part][0];
                const point_2 = body_parts[body_part][1];
                if (points[point_1] && points[point_2])
                    measures[body_part.split('_')[1]].push(vector_length(points[point_1], points[point_2]));
            }

            setTimeout(callback, 5);

        });

    }
}

function finalize(callback) {

    let real_measures = {};

    for (const k in measures) {
        if (measures[k].length > 0)
            measures[k] = measures[k].reduce(function(a, b) { return a + b; }) / measures[k].length;
        else
            measures[k] = 'N/A';
    }

    const adjusted = (measures["box_width"] / real_size[0] + measures["box_height"] / real_size[1]) / 2;

    const slopes = {
        "arm": 1.0085161, "forearm": 1.1133968,
        "leg": 1.66024735, "thigh": 1.34745464,
        "shoulders": 1.25277162, "trunk": 0.96818675, "pelvis": 1.77853941
    };

    for (const k in measures) {
        if (k != 'box_width' && k != 'box_height') {
            if (measures[k] != 'N/A') real_measures[k] = (measures[k] / adjusted) * slopes[k];
            else real_measures[k] = 'N/A';
        }
    }

    callback(real_measures);

}
