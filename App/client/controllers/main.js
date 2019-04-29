import {loading_models, process_image, start, stop} from "./scan";

let recording = true;

Template.main.created = function () {
    Session.set("processing_process", 0);
    Session.set("size_results", "");
    //console.log('GOO');
    loading_models(model_loaded);
    //model_loaded();
};

Template.main.rendered = function () {

    let options = {
        x: 0, y: 0,
        width: window.screen.width,
        height: window.screen.height,
        camera: CameraPreview.CAMERA_DIRECTION.BACK,
        toBack: true,
        tapPhoto: false,
        tapFocus: false,
        previewDrag: false
    };

    CameraPreview.startCamera(options);
    CameraPreview.setFocusMode(CameraPreview.FOCUS_MODE.CONTINUOUS_PICTURE);
    recording = false;

};

Template.main.helpers({
   'processing_progress'() {
       return Session.get("processing_process");
   },
   'size_results'() {
       return Session.get('size_results');
   }
});

Template.main.events({
   "click #close_results"() {
     $("#results").css('display', 'none');
   },
   "click #scanning"() {
       if ($("#scanning").hasClass('active')) {
           $("#scanning").removeClass('active');
           $("#scanning").removeClass('blue');
           $("#scanning").addClass('red');
           $("#scanning i").html("camera");

           recording = false;

       } else {
           $("#scanning").removeClass('red');
           $("#scanning").addClass('blue');
           $("#scanning").addClass('active');
           $("#scanning i").html("stop");
           $('body').addClass('laser');

           recording = true;

           $('#looking_for').css('display', 'block');
           $('#scanning_for').css('display', 'none');

           start();
           record(process_callback);
       }
   }
});

function model_loaded() {
    record(function () { $("#loading").remove(); });

}

function display_progress(progress) {
    Session.set("processing_process", progress);
}

function display_results(results) {
    $("#processing").css('display', 'none');
    $("#results").css('display', 'block');
    $("#scanning").css('display', '');
    const results_list = [];
    const keys = ['arm', 'forearm', 'thigh', 'leg', 'trunk', 'shoulders', 'pelvis'];

    for (const k in keys)
        results_list.push({'key': keys[k], 'value': results[keys[k]] != 'N/A' ? Math.round(results[keys[k]]) : results[keys[k]]});

    Session.set("size_results", results_list);
}

function process_callback(found) {

    if (found) {
        $('#looking_for').css('display', 'none');
        $('#scanning_for').css('display', 'block');
    }

    const waiting_time = found ? 500 : 10;

    if (recording) setTimeout(function () { record(process_callback); }, waiting_time);
    else {
        $('#looking_for').css('display', 'none');
        $('#scanning_for').css('display', 'none');
        if (found) {
            $("#processing").css('display', 'block');
            $("#scanning").css('display', 'none');
            stop(display_progress, display_results);
        }
    }
}

function record(callback) {

    CameraPreview.takePicture({width: 240, height: 320, quality: 85}, function(base64PictureData) {

        const image = new Image();
        image.src = 'data:image/jpeg;base64,' + base64PictureData;

        image.onload = function() { console.log(image.width); console.log(image.height); process_image(image, callback); };

    });

}