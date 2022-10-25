# A Simple HTTP Server with YOLACT
Sometimes, we need a REST API to get the detected object data for the instance segmentation. Since YOLACT is a powerful model for the real-time segmentation, I implement the simple HTTP server for it. This project will return the response that contains the meta data of detected objects from single frame while the client sends the full path of image. 

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/JamesWanglf/yolact-instance-segmentation.git
   cd yolact-instance-segmentation
   ```
   This project is based on Yolact project. For the detailed installation, please read https://github.com/dbolya/yolact.
 - Run HTTP server:
	```Shell
	python3 rest_api_server.py
	```
	This will run the simple HTTP service on 6337 port at the localhost.
	It will support the following URLs to access.
	- http://localhost:6337/detect_objects?image_path=[image_full_path]&top_k=[object_amount]&score_threshold=[object_acuraty]
		In this url, you can send the following parameters.
		- image_path
			This is the full path of the single frame. This parameter is required essentially.
		- top_k
			the maximum amount of the detected objects what you want to get. This is an optional parameter. If you don't set this parameter, it will be set by 10 automatically.
		- score_threshold
			the minumum acuraty of the detected objects. API will only return the meta data of objects whose acuraty is over than this value.
	- http://localhost:6337/init_engine?model=[model_name]
		You can init the model by model name. You can call the /detect_objects api directly without /init_engine, the API will use the "yolact_resnet50_54_800000.pth" as default. But in some use cases, you might need to change the model for your project. In that case, you can use this url. 
		One thing you need to pay attention is that you must put the model file into ./weights folder before you call this url.
