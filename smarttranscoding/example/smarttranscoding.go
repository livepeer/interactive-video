package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"strings"

	"github.com/gorilla/websocket"
	"github.com/livepeer/interactive-video/smarttranscoding/ffmpeg"
)

var addr = flag.String("addr", "localhost:8080", "http service address")

var nodes = []string{"127.0.0.1", "35.194.58.82", "34.135.170.77"}

// var nodes = []string{"127.0.0.1"}

type Client struct {
	ID   string
	Conn *websocket.Conn
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

var clients map[*websocket.Conn]bool

var isFirstFrame = true

func handleconnections1(w http.ResponseWriter, r *http.Request) {

	codec := r.Header.Get("X-WS-Codec")
	size := r.Header.Get("X-WS-Video-Size")
	sizetmp := strings.Split(size, "x")
	width, _ := strconv.Atoi(sizetmp[0])
	height, _ := strconv.Atoi(sizetmp[1])

	respheader := make(http.Header)
	initData := r.Header.Get("X-Ws-Init")
	spsData, _ := base64.StdEncoding.DecodeString(initData)

	// var instances []ffmpeg.Instance

	facebuf1, _ := ioutil.ReadFile("example/faces/face1")
	facebuf2, _ := ioutil.ReadFile("example/faces/face2")
	facebuf3, _ := ioutil.ReadFile("example/faces/philipp")
	facebuf4, _ := ioutil.ReadFile("example/faces/eric")

	metadata := fmt.Sprintf(`
	[
		{"id": "1", "name": "Nick", "image": "%s", "metadata": "https://livepeer.org/dev1", "action": "embedlink"}, 
		{"id": "2", "name": "James", "image": "%s", "metadata": "https://livepeer.org/dev2", "action": "embedlink"},
		{"id": "3", "name": "Philipp", "image": "%s", "metadata": "https://livepeer.org/Philipp", "action": "embedlink"},
		{"id": "4", "name": "Eric", "image": "%s", "metadata": "https://livepeer.org/Eric", "action": "embedlink"}
	]
	`, string(facebuf1), string(facebuf2), string(facebuf3), string(facebuf4))

	resp, err := ffmpeg.RegisterSamples(bytes.NewBuffer([]byte(metadata)))

	if err != nil {
		log.Print("instance registration failure", err)
	} else {
		respBody, _ := io.ReadAll(resp.Body)
		fmt.Println(string(respBody))
	}

	respheader.Add("Sec-WebSocket-Protocol", "videoprocessing.livepeer.com")
	c, err := upgrader.Upgrade(w, r, respheader)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()

	fmt.Println("video codec id:", codec, width, height)

	ffmpeg.SetDecoderCtxParams(width, height)
	handlemsg1(w, r, c, codec, spsData)

}

func handlemsg1(w http.ResponseWriter, r *http.Request, conn *websocket.Conn, codec string, initData []byte) {
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("error: %v", err)
			}
			log.Printf("read:%v", err)
			conn.Close()
			isFirstFrame = true
			break
		}
		timestamp := binary.BigEndian.Uint64(message[:8])
		packetdata := message[8:]

		if isFirstFrame {
			fmt.Println("sps packet, appending initData", initData)
			packetdata = append(initData, packetdata...)
			isFirstFrame = false
		}

		timedpacket := ffmpeg.TimedPacket{Timestamp: timestamp, Packetdata: ffmpeg.APacket{Data: packetdata, Length: len(packetdata)}}
		ffmpeg.FeedPacket(timedpacket, nodes, conn, nodes)
	}
}

func startServer1() {
	log.Println("started server", *addr)
	http.HandleFunc("/segmentation", handleconnections1)
}

func main() {
	flag.Parse()
	log.SetFlags(0)
	ffmpeg.DecoderInit()
	startServer1()
	log.Fatal(http.ListenAndServe(*addr, nil))
}
