var waiting = false
var connected = false
var ws
var reconnectTimeout

//--- Model, Singleton

const Store = {
	data: {
		"image":"",
		"controls": {
			"angle":0,
			"yaw":0,
			"throttle":0
		},
		"auto_pilot": {
			"auto_pilots":["None"],
			"index":0
		},
		"sensor_reading": {
			"sonar_left":0,
			"sonar_right":0,
			"m9ax":0,
			"m9ay":0,
			"m9az":0,
			"m9gx":0,
			"m9gy":0,
			"m9gz":0,
			"m9mx":0,
			"m9my":0,
			"m9mz":0
		},
		"fps":0,
		"record": false,
		"throttle": false,
		"is_recording": false,
		"f_time": 0
	},
	updateData: function(data) {
		const dataCopy = Object.assign({}, this.data)
		for (let attrname in data) { dataCopy[attrname] = data[attrname] }
		this.data = dataCopy
	}
}

//--- Dispatcher, Singleton

const Dispatcher = {
	set: function(payload, update_backend) {
		Store.updateData(payload.value)
		if (update_backend) {
			payload = Object.assign({}, payload)
			denormalizeOutgoingPayload(payload.value)
			payload.action = "set"
			ws.send(JSON.stringify(payload))
			waiting = true
		}
		if (waiting == false) {
			m.redraw()
		}
	}
}

//--- API, Singleton

function connect() {
	var host = window.location.host;
	ws = new WebSocket("ws://"+host+"/api/v1/ws")
  	ws.onmessage = function (event) {
		const obj = JSON.parse(event.data)
		if (obj.ack == "ok") {
			waiting = false
			m.redraw()
		}
		else if (obj.image) {
			const payload = {"target": "data", "value": obj}
			normalizeIncomingPayload(payload)
			Dispatcher.set(payload, false)
		    setTimeout(function() {
		    	const payload = {"target": "status", "action": "get"}
		    	ws.send(JSON.stringify(payload))
		    }, 100)
		}
		connected = true
		reconnectTimeout = 1000
	}
	ws.onclose = function(e) {
		console.log('Socket is closed. Reconnect will be attempted in '+reconnectTimeout+' ms.', e.reason)
		connected = false
		m.redraw()
		setTimeout(function() { connect() }, reconnectTimeout)
		reconnectTimeout = Math.min(30000, Math.round(reconnectTimeout * 1.2))
	}
	ws.onerror = function(err) {
		console.error('Socket encountered error: ', err.message, 'Closing socket')
		ws.close()
	}
}

connect()

//--- Views, Factory

const ImageView =  {
	view: function() {
		return m("img", {class:"viewport", src:"data:image/jpeg;base64," + Store.data.image})
	}
}

const CommandView = {
	view: function() {
		const classes = classNames({
		    sliderBox: true,
		    record_enable: Store.data.record,
		    recording: Store.data.is_recording
		})
		const left = Math.min(Math.max(Store.data.controls.yaw, -1), 1) * 50 + 50
		return m( "div", {class: classes},
			m("div", {class:"sliderKnob", style:"left:" + left + "%;"})
		)
	}
}

const PilotsView = {
	view: function (ctrl) {
	    return m('select', {
	    	onchange: m.withAttr('value', function(value) {
	    		const payload = {"target": "auto_pilot",
	    						 "value": {"index" : Store.data.auto_pilot.auto_pilots.indexOf(value)}}
		    	Dispatcher.set(payload, true)
	    	}
	    ) }, [
	      	Store.data.auto_pilot.auto_pilots.map(function(name, index) {
	        	return m('option' + (Store.data.auto_pilot.index === index  ? '[selected=true]' : ''), name)
	      	})
	    ])
	}
}

const RecordBox = {
	view: function(ctrl) {
		return m('input', {
			type: "checkbox",
			class: "js-switch",
			checked: Store.data.record,
			onchange: m.withAttr('checked', function(checked) {
				const payload = {"target": "record", "value": {"record" : checked}}
		    	Dispatcher.set(payload, true)
			}) }, "Record")
	}
}

const ThrottleBox = {
	view: function(ctrl) {
		return m('input', {
			type: "checkbox",
			class: "js-switch",
			checked: Store.data.throttle,
			onchange: m.withAttr('checked', function(checked) {
				const payload = {"target": "throttle", "value": {"throttle" : checked}}
		    	Dispatcher.set(payload, true)
			}) }, "Throttle")
	}
}

const ThrottleValue = {
	view: function() {
		return m("span", Math.round((parseFloat(Store.data.controls.throttle) + 0.00001) * 100) / 100)
	}
}

const FTimeValue = {
	view: function() {
		return m("span", Math.round((Store.data.f_time + 0.00001) * 1000) + " ms")
	}
}

const ConnectionLED = {
	view: function() {
		return m("div",
			[m("span", {id:"connectionLED", class: connected ? "green" : "red"}),
			 m("span", {id:"connectionCaption"}, connected ? "Connected" : "Not Connected")]
		)
	}
}

const SonarLeftValue = {
	view: function() {
		return m("span", (Math.round((Store.data.sensor_reading.sonar_left_ft + 0.00001) * 100) / 100).toFixed(2)+" ft")
	}
}

const FPSValue = {
	view: function() {
		return m("span", Store.data.fps.toFixed(2))
	}
}

const SonarRightValue = {
	view: function() {
		return m("span", (Math.round((Store.data.sensor_reading.sonar_right_ft + 0.00001) * 100) / 100).toFixed(2)+" ft")
	}
}

const AccelerometerValue = {
	view: function() {
		return m("span", (Math.round((Store.data.sensor_reading.m9ax + 0.00001) * 100) / 100).toFixed(2)+" "+
			(Math.round((Store.data.sensor_reading.m9ay + 0.00001) * 100) / 100).toFixed(2)+" "+
			(Math.round((Store.data.sensor_reading.m9az + 0.00001) * 100) / 100).toFixed(2))
	}
}

const GyroscopeValue = {
	view: function() {
		return m("span", (Math.round((Store.data.sensor_reading.m9gx + 0.00001) * 100) / 100).toFixed(2)+" "+
			(Math.round((Store.data.sensor_reading.m9gy + 0.00001) * 100) / 100).toFixed(2)+" "+
			(Math.round((Store.data.sensor_reading.m9gz + 0.00001) * 100) / 100).toFixed(2))
	}
}

const MagnetometerValue = {
	view: function() {
		return m("span", (Math.round((Store.data.sensor_reading.m9mx + 0.00001) * 100) / 100)+" "+
			(Math.round((Store.data.sensor_reading.m9my + 0.00001) * 100) / 100)+" "+
			(Math.round((Store.data.sensor_reading.m9mz + 0.00001) * 100) / 100))
	}
}

const Veil = {
	view: function(ctrl) {
		const vis = waiting == true?"visible":"hidden"
		const style = "visibility:" + vis + ";"
		return m('div', {class:"veil", style:style},"")
	}
}

m.mount(document.getElementById("imageContainer"), ImageView)
m.mount(document.getElementById("sliderContainer"), CommandView)
m.mount(document.getElementById("pilotsContainer"), PilotsView)
m.mount(document.getElementById("throttleBox"), ThrottleBox)
m.mount(document.getElementById("recordBox"), RecordBox)
m.mount(document.getElementById("throttleValue"), ThrottleValue)
m.mount(document.getElementById("ftimeValue"), FTimeValue)
m.mount(document.getElementById("connectionLEDContainer"), ConnectionLED)
m.mount(document.getElementById("veilContainer"), Veil)
m.mount(document.getElementById("sonarLeft"),SonarLeftValue)
m.mount(document.getElementById("sonarRight"),SonarRightValue)
m.mount(document.getElementById("accelerometer"),AccelerometerValue)
m.mount(document.getElementById("gyroscope"),GyroscopeValue)
m.mount(document.getElementById("magnetometer"),MagnetometerValue)
m.mount(document.getElementById("fps"),FPSValue)



const draggie = new Draggabilly('#primaryContainer', { containment: '#container' })

window.onresize = function(event) {
    if (document.documentElement.clientWidth <= 920) {
    	draggie.disable()
    } else {
    	draggie.enable()
    }
}

function normalizeIncomingPayload(payload) {
	payload.value.auto_pilot.auto_pilots.unshift("None")
	payload.value.auto_pilot.index += 1
}

function denormalizeOutgoingPayload(payload) {
	if (payload.hasOwnProperty("index")) {
		payload.index -= 1
	}
}
