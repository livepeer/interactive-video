package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/livepeer/interactive-video/liveasr/ffmpeg"
	webrtcvad "github.com/maxhawkins/go-webrtcvad"
)

var addr = flag.String("addr", "localhost:8080", "http service address")
var duration = flag.Uint64("duration", 6000, "segment duration")
var m sync.Mutex

// var numOfsubsegs = flag.Uint64("segcount", 3, "number of subsegments")

type Client struct {
	ID   string
	Conn *websocket.Conn
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

var isFirstFrame = true
var prevTimestamp uint64

var transcribeUrl = "http://127.0.0.1:5000/transcribe"

func handleconnections1(w http.ResponseWriter, r *http.Request) {

	codec := r.Header.Get("X-WS-Audio-Codec")
	channel := r.Header.Get("X-WS-Audio-Channels")
	sample_rate := r.Header.Get("X-WS-Rate")
	bit_rate := r.Header.Get("X-WS-BitRate")

	if codec == "" || channel == "" || sample_rate == "" || bit_rate == "" {
		log.Print("audio meta data not present in header, handshake failed.")
		return
	}
	respheader := make(http.Header)
	respheader.Add("Sec-WebSocket-Protocol", "speechtotext.livepeer.com")

	c, err := upgrader.Upgrade(w, r, respheader)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()

	fmt.Println("video codec id:", codec, sample_rate)

	// ffmpeg.SetDecoderCtxParams(width, height)
	handlemsg1(w, r, c, codec)

}

func handlemsg1(w http.ResponseWriter, r *http.Request, conn *websocket.Conn, codec string) {
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
		curTimestamp := binary.BigEndian.Uint64(message[:8])
		if isFirstFrame {
			prevTimestamp = curTimestamp
			isFirstFrame = false
		}
		packetdata := message[8:]
		filename := fmt.Sprintf("temp%d.aac", prevTimestamp)
		f, err := os.OpenFile(filename, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0600)
		if err != nil {
			panic(err)
		}

		defer f.Close()

		if _, err = f.Write(packetdata); err != nil {
			panic(err)
		}

		vad, err := webrtcvad.New()
		if err != nil {
			log.Fatal(err)
		}

		if err := vad.SetMode(2); err != nil {
			log.Fatal(err)
		}

		tempTimestamp := prevTimestamp
		if (curTimestamp - prevTimestamp) >= *duration {
			fmt.Println("splitting audio")
			go ffmpeg.FeedPacket(filename, conn, transcribeUrl, curTimestamp-500, *duration-100, &m)
			// if (curTimestamp - prevTimestamp) >= *duration {
			prevTimestamp = curTimestamp
			// }
		} else if curTimestamp-tempTimestamp >= 500 {
			res := map[string]interface{}{"timestamp": curTimestamp - 500, "text": "", "duration": *duration - 100}
			jsonres, _ := json.Marshal(res)
			m.Lock()
			conn.WriteMessage(websocket.TextMessage, []byte(string(jsonres)))
			m.Unlock()
			tempTimestamp = curTimestamp
		}
	}
}

func startServer1() {
	log.Println("started server", *addr)
	http.HandleFunc("/speech2text", handleconnections1)
}

func main() {
	flag.Parse()
	log.SetFlags(0)
	// ffmpeg.DecoderInit()
	ffmpeg.LoadModel()
	startServer1()
	log.Fatal(http.ListenAndServe(*addr, nil))
}
