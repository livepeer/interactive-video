# Interactive-video

Interactive video provides interactive video experience both to stream viewers and broadcasters by leveraging AI in video/audio understanding. By leveraging the power of community of GPU nodes, it aims to provide configurable, selectable AI features.

## System architecture
![](img/smart%20av%20architecture.png)

The system consists of 4 parts at large:
* API Server
* AI nodes
* Media Server (MistServer + Interactive Video UI)
* Feedback Server (for broadcasters to monitor user interactions/statistics)

## Workflow
1. Mistserver sends source video + SmartVideo requests to Orchestrator over websocket. The smart video request parameters (object detection, face recognition) are set at websocket connection initialization.
2. Orchestrator decodes timestamped video packets and outsource jobs to AI nodes. Job distribution is by job type and also in time, meaning same different jobs are distributed to multiple nodes in time domain. For example, frame1 is processed by facerecognition_node1, frame2 is by facerecognition_node2, etc etc. The reason for doing this is to make sure that real time factor > 1 so that meta stream does not fall behind the video stream.
3. AI nodes process decoded frames(images) and return results to Orchestrator. The metadata returned by AI nodes are aggregated at Orchestrator.
4. Orchestrator sends metadata (aggregated AI results) to MistServer + Web UI over websocket.
5. The Web UI receives the metadata stream from MistServer and displays the metadata and hardcoded action set on the right sidebar of the Web UI. Users are allowed to click on metadata and hardcoded action set on the Web UI within limited timewindow.
6. The interaction results are sent to the feedback server over http. They are sent at either user click send button or at timewindow timeout.


## Installation
Installation guide can be found at Readme files of each sub project.