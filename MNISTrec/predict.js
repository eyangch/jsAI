const imgX = 28;
const imgY = 28;

const canvX = 280;
const canvY = 280;

const strokewidth = 2;

var canvimgX = canvX/imgX;
var canvimgY = canvY/imgY;

function onEpochEnd(epoch, logs){
	document.getElementById("epoch").innerHTML = "Epoch: " + epoch;
}

function onBatchEnd(batch, logs){
	console.log("Batch: " + batch + " Loss: " + logs.loss + " Acc " + logs.acc);
	document.getElementById("statusBatch").innerHTML = "Batch: " + batch + " Loss: " + logs.loss + " Acc " + logs.acc;
}


class AImodel{
	constructor(){	
		this.model = tf.sequential();
	}
	async init(){
		this.model = await tf.loadLayersModel("https://eyangch.github.io/jsAI/MNISTrec/model1.json");
		return this.model;
	}
	initAux(){
		this.model.compile({optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"]});
		this.model.summary();
	}
	predictModel(inp){
		inp = tf.tensor(inp).reshape([-1, imgX, imgY, 1]);
		return this.model.predict(inp).arraySync();
	}
	async saveModel(){
		console.log("SAVE MODEL");
		document.getElementById("output").innerHTML = "Downloading Model";
		return await this.model.save("downloads://model1");

	}
}

var trainData = [];
var trainLabels = [];
var classifier = new AImodel();
classifier.init().then(function(){classifier.initAux()});

var drawing = false;

function clearcanvas(){
	var canvtx = document.getElementById("draw").getContext("2d");
	var canvtx2 = document.getElementById("op").getContext("2d");
	canvtx.clearRect(0, 0, canvX, canvY);
	canvtx2.clearRect(0, 0, imgX, imgX);
}

function predict(){
	var canvtx2 = document.getElementById("op").getContext("2d");
	var imgData = canvtx2.getImageData(0, 0, imgX, imgY).data;
	var i1 = 0;
	for(var i = 3; i < imgData.length; i += 4){
		imgData[i1] = imgData[i] / 255;
		i1++;
	}
	imgData = imgData.slice(0, imgData.length/4);
	var pData = [];
	for(var i = 0; i < imgY; i++){
		pData.push(Array.prototype.slice.call(imgData.slice(i * imgX, (i + 1) * imgX)));
	}
	var pred = classifier.predictModel(pData);
	document.getElementById("pred").innerHTML = "";
	document.getElementById("pred2").innerHTML = "";
	var maxInd = 0;
	var maxNum = 0;
	for(var i = 0; i < pred[0].length; i++){
		document.getElementById("pred").innerHTML += i.toString() + " probability: " + pred[0][i].toFixed(8) + "\n";
		if(pred[0][i] > maxNum){
			maxNum = pred[0][i];
			maxInd = i;
		}
	}
	document.getElementById("pred2").innerHTML += "The AI thinks the number you drew is <strong>" + maxInd.toString() + "</strong>\n";
	var closeEnough = maxNum/10;
	var runnerups = [];
	for(var i = 0; i < pred[0].length; i++){
		if(pred[0][i] >= closeEnough && i != maxInd){
			runnerups.push(i);
		}
	}
	var comma = " ";
	if(runnerups.length > 2){
		comma = ", ";
	}
	if(runnerups.length == 1){
		document.getElementById("pred2").innerHTML += "The AI also thinks the number you drew could be <strong>" + runnerups[0].toString() + "</strong>\n";
	}else if(runnerups.length > 1){
		document.getElementById("pred2").innerHTML += "The AI also thinks the number you drew could be ";
		for(var i = 0; i < runnerups.length - 1; i++){
			document.getElementById("pred2").innerHTML += "<strong>" + runnerups[i].toString() + "</strong>" + comma;
		}
		document.getElementById("pred2").innerHTML += "or <strong>" + runnerups[i].toString() + "</strong>" + "\n";
	}
}

function drawPoints(canv, canv2, event){
	if(drawing){
		canvB = canv.getBoundingClientRect();
		var x = event.clientX - canvB.left;
		var y = event.clientY - canvB.top;
		var xadj = Math.floor(x / canvimgX - strokewidth/2);
		var yadj = Math.floor(y / canvimgY - strokewidth/2);
		var drawC1 = canv.getContext("2d");
		var drawC2 = canv2.getContext("2d");
		drawC1.fillRect(xadj * canvimgX, yadj * canvimgY, canvimgX * strokewidth, canvimgY * strokewidth);
		drawC2.fillRect(xadj, yadj, strokewidth, strokewidth);
	}
}

window.onload = function(){
	var canv = document.getElementById("draw");
	var canv2 = document.getElementById("op");
	canv.addEventListener("mousemove", function(event) {
		drawPoints(canv, canv2, event);
	});
	canv.addEventListener("touchmove", function(event) {
		drawPoints(canv, canv2, event.touches[0]);
	});
	canv.addEventListener("mousedown", function(event) {
		drawing = true;
		drawPoints(canv, canv2, event);
	});
	canv.addEventListener("touchstart", function(event) {
		drawing = true;
		drawPoints(canv, canv2, event.touches[0]);
	});
	document.addEventListener("mousedown", function(event) {
		drawing = true;
	});
	document.addEventListener("mouseup", function(event) {
		drawing = false;
	});
	document.addEventListener("touchstart", function(event) {
		drawing = true;
	});
	document.addEventListener("touchend", function(event) {
		drawing = false;
	});
}
