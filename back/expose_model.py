
import flask
import subprocess

app = flask.Flask('model_api')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', "OSLC-Core-Version, Access-Control-Allow-Origin, Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers")
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    return response


@app.route('/model/<d>/file.pb', methods=['GET', 'OPTIONS'])
def model(d):
    return flask.send_file('./web_model/tensorflowjs_model.pb')


@app.route('/model_weights/<d>/file.json', methods=['GET', 'OPTIONS'])
def model_weights(d):
    return flask.send_file('./web_model/weights_manifest.json')


@app.route('/model_weights/<d>/<name>', methods=['GET', 'OPTIONS'])
def model_group1shard1of1(d, name):
    return flask.send_file('./web_model/' + name)


app.run('0.0.0.0', 1997)
