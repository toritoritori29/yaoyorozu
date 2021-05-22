



var express = require("express");
var app = express();

app.use('/dist', express.static(__dirname + '/dist'));
app.use('/models', express.static(__dirname + '/models'));

app.get('/', (req, res) => {
  console.log(__dirname);
  res.sendFile(__dirname + '/index.html');
});

const PORT_NUM = 8080;
var server = app.listen(PORT_NUM, () => {
  console.log(`Starting up yaoyorozu-example-server, listen to ${PORT_NUM}`);
});