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
	"path/filepath"
	"strconv"
	"sync"
	"unsafe"

	"github.com/gorilla/websocket"
)

// #cgo pkg-config: libavformat libavfilter libavcodec libavutil libswscale gnutls
// #include <stdlib.h>
// #include "lpms_ffmpeg.h"
import "C"

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

func DecoderInit() {
	C.video_codec_init()
	fmt.Println("decoder initialized")
}

func SetDecoderCtxParams(w int, h int) {
	C.set_decoder_ctx_params(C.int(w), C.int(h))
}

var m sync.Mutex

func FaceRecognition(filename string, conn *websocket.Conn, url string, timestamp uint64, m *sync.Mutex) {
	f, _ := os.Open(filename)
	defer os.Remove(filename)
	retStr := ""
	client := &http.Client{}
	// Read entire JPG into byte slice.
	reader := bufio.NewReader(f)
	content, _ := ioutil.ReadAll(reader)

	// Encode as base64.
	encoded := base64.StdEncoding.EncodeToString(content)

	// Print encoded data to console.
	// ... The base64 image can be used as a data URI in a browser.
	// fmt.Println("ENCODED: " + encoded)
	metadata := fmt.Sprintf(`{"image": "data:image/jpg;base64, %s"}`, encoded)
	// fmt.Println(encoded, "\n\n\n\n")
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

	res := map[string]interface{}{"track": "face-recognition", "timestamp": int(timestamp), "metadata": retStr, "type": "metadata"}
	jsonres, _ := json.Marshal(res)
	log.Println("writing data, timestamp:", timestamp, "\n", string(jsonres))
	m.Lock()
	conn.WriteMessage(websocket.TextMessage, []byte(string(jsonres)))
	m.Unlock()
}

func ImageCaptioning(filename string, conn *websocket.Conn, url string, timestamp uint64, m *sync.Mutex) {
	f, _ := os.Open(filename)
	defer os.Remove(filename)
	retStr := ""
	client := &http.Client{}

	// Read entire JPG into byte slice.
	reader := bufio.NewReader(f)
	content, _ := ioutil.ReadAll(reader)

	// Encode as base64.
	encoded := base64.StdEncoding.EncodeToString(content)
	metadata := fmt.Sprintf(`{"image": "data:image/jpg;base64, %s"}`, encoded)
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

	res := map[string]interface{}{"track": "image-caption", "timestamp": int(timestamp), "metadata": retStr, "type": "metadata"}
	jsonres, _ := json.Marshal(res)
	log.Println("writing data, timestamp:", timestamp, "\n", string(jsonres))
	m.Lock()
	conn.WriteMessage(websocket.TextMessage, []byte(string(jsonres)))
	m.Unlock()
}

func InstanceSegmentation(filename string, conn *websocket.Conn, url string, timestamp uint64, m *sync.Mutex) {
	f, _ := os.Open(filename)
	defer os.Remove(filename)
	retStr := ""
	client := &http.Client{}

	// Read entire JPG into byte slice.
	reader := bufio.NewReader(f)
	content, _ := ioutil.ReadAll(reader)

	// Encode as base64.
	encoded := base64.StdEncoding.EncodeToString(content)

	// Print encoded data to console.
	// ... The base64 image can be used as a data URI in a browser.
	// fmt.Println("ENCODED: " + encoded)
	metadata := fmt.Sprintf(`{"image": "data:image/jpg;base64, %s"}`, encoded)
	// fmt.Println(encoded, "\n\n\n\n")
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer([]byte(metadata)))
	if err != nil {
		log.Fatal(err)
	}
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
	} else {
		log.Printf("url: %s\n status code %d", url, resp.StatusCode)
	}

	res := map[string]interface{}{"track": "instance-segmentation", "timestamp": int(timestamp), "metadata": retStr, "type": "metadata"}
	jsonres, _ := json.Marshal(res)
	log.Println("writing mjpg data, timestamp:", timestamp, " node:", url)
	m.Lock()
	conn.WriteMessage(websocket.TextMessage, []byte(string(jsonres)))
	m.Unlock()
}

func FeedPacket(pkt TimedPacket, nodes []string, conn *websocket.Conn, reqFeatures []string) {
	timestamp := pkt.Timestamp
	pktdata := pkt.Packetdata
	buffer := (*C.char)(unsafe.Pointer(C.CString(string(pktdata.Data))))
	defer C.free(unsafe.Pointer(buffer))
	C.ds_feedpkt(buffer, C.int(pktdata.Length), C.int(pkt.Timestamp))

	path, _ := os.Getwd()
	filename := filepath.Join(path, "frame"+strconv.Itoa(int(pkt.Timestamp))+".jpg")

	url := ""
	nodelen := len(nodes)

	if i%5 == 0 {
		url = fmt.Sprintf("http://%s:5000/face-recognition", nodes[i%nodelen])
		go FaceRecognition(filename, conn, url, timestamp, &m)
	} else if i%30 == 1 {
		url = fmt.Sprintf("http://%s:5000/image-captioning", nodes[i%nodelen])
		go ImageCaptioning(filename, conn, url, timestamp, &m)
	} else if i%10 == 2 {
		url = fmt.Sprintf("http://%s:5000/instance-segmentation/detect-objects", nodes[i%nodelen])
		go InstanceSegmentation(filename, conn, url, timestamp, &m)
	} else {
		defer os.Remove(filename)
	}
	i = (i + 1) % 30
}

func RegisterSamples(registerData *bytes.Buffer) (*http.Response, error) {
	client := &http.Client{}

	url := "http://127.0.0.1:5000/face-recognition/update-samples"

	req, _ := http.NewRequest("POST", url, registerData)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	fmt.Println("response Status:", resp.Status)
	fmt.Println("response Headers:", resp.Header)
	body, _ := ioutil.ReadAll(resp.Body)
	fmt.Println("response Body:", string(body))
	return resp, err
}
