package ffmpeg

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/gorilla/websocket"
)

// #cgo pkg-config: libavformat libavfilter libavcodec libavutil libswscale gnutls
// #include <stdlib.h>
// #include "lpms_ffmpeg.h"
// import "C"

type TimedPacket struct {
	Packetdata APacket
	Timestamp  uint64
}

type APacket struct {
	Data   []byte
	Length int
}

type Instance struct {
	Id       string `json:"id"`
	Name     string `json:"name"`
	Image    string `json:"image"`
	MetaData string `json:"metadata"`
	Action   string `json:"action"`
}

var i int = 0

// func DecoderInit() {
// 	C.video_codec_init()
// 	fmt.Println("decoder initialized")
// }

func LoadModel() {
	client := &http.Client{}

	url := "http://127.0.0.1:5000/load_model"

	postdata := `{
		"language": "en",
		"model_size": "base"
	}`

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer([]byte(postdata)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		fmt.Println("model loaded successfully")
	}
}

// func SetDecoderCtxParams(w int, h int) {
// 	C.set_decoder_ctx_params(C.int(w), C.int(h))
// }

func Transcribe(filename string, conn *websocket.Conn, url string, timestamp uint64, duration uint64, m *sync.Mutex) {
	f, _ := os.Open(filename)
	defer os.Remove(filename)
	retStr := ""
	client := &http.Client{}
	// Read entire file into byte slice.
	reader := bufio.NewReader(f)
	content, _ := ioutil.ReadAll(reader)

	// Encode as base64.
	encoded := base64.StdEncoding.EncodeToString(content)

	metadata := fmt.Sprintf(`{"file_ext": "aac",  "file_content": "%s"}`, encoded)

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer([]byte(metadata)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Fatal(err)
		}
		bodyString := string(bodyBytes)
		retStr = bodyString
	}

	res := map[string]interface{}{"timestamp": timestamp, "duration": duration, "text": retStr}

	jsonres, _ := json.Marshal(res)
	log.Println("writing data, timestamp:", timestamp, "\n", string(jsonres))
	m.Lock()
	conn.WriteMessage(websocket.TextMessage, []byte(string(jsonres)))
	m.Unlock()
}

func FeedPacket(filename string, conn *websocket.Conn, url string, timestamp uint64, duration uint64, m *sync.Mutex) {
	go Transcribe(filename, conn, url, timestamp, duration, m)
}
