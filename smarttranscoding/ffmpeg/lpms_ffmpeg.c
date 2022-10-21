#include "lpms_ffmpeg.h"

#include <libavcodec/avcodec.h>

#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#include <pthread.h>
#include <unistd.h>

#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"

AVCodec *decoder = NULL;
AVCodecContext *decoder_ctx = NULL;
AVCodecParserContext *parser = NULL;

static AVBufferRef *hw_device_ctx = NULL;
static enum AVPixelFormat hw_pix_fmt;
enum AVHWDeviceType device_type;

static int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type)
{
    int err = 0;

    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
                                      NULL, NULL, 0)) < 0) {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    return err;
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                        const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

static void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame)
{
    FILE *pFile;
    char szFilename[32];
    int  y;

    // Open file
    sprintf(szFilename, "frame%d.ppm", iFrame);
    pFile=fopen(szFilename, "wb");
    if(pFile==NULL)
        return;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for(y=0; y<height; y++)
        fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);

    // Close file
    fclose(pFile);
}



// int writeJPEG(AVFrame* frame,int width,int height, int iFrame){
//     char out_file[32];
//     sprintf(out_file, "frame%d.jpg", iFrame);
//     //新建一个输出的AVFormatContext 并分配内存
//     AVFormatContext* output_cxt = avformat_alloc_context();
//     avformat_alloc_output_context2(&output_cxt,NULL,"singlejpeg",out_file);

//     //设置输出文件的格式
//     // output_cxt->oformat = av_guess_format("mjpeg",NULL,NULL);

//     //创建和初始化一个和该URL相关的AVIOContext
//     if(avio_open(&output_cxt->pb,out_file,AVIO_FLAG_READ_WRITE) < 0){
//         av_log(NULL,AV_LOG_ERROR,"cannot open file\n");
//         return -1;
//     }

//     //构建新的Stream
//     AVStream* stream = avformat_new_stream(output_cxt,NULL);
//     if(stream == NULL){
//         av_log(NULL,AV_LOG_ERROR,"failed to create AVStream\n");
//         return -1;
//     }
//     //初始化AVStream信息
//     AVCodecContext* codec_cxt = stream->codec;

//     codec_cxt->codec_id = output_cxt->oformat->video_codec;
//     codec_cxt->codec_type = AVMEDIA_TYPE_VIDEO;
//     codec_cxt->pix_fmt = AV_PIX_FMT_YUVJ420P;
//     codec_cxt->height = height;
//     codec_cxt->width = width;
//     codec_cxt->time_base.num = 1;
//     codec_cxt->time_base.den = 25;

//     //打印输出文件信息
//     av_dump_format(output_cxt,0,out_file,1);

//     AVCodec* codec = avcodec_find_encoder(codec_cxt->codec_id);
//     if(!codec){
//         av_log(NULL,AV_LOG_ERROR,"cannot find encoder\n");
//         return -1;
//     }

//     if(avcodec_open2(codec_cxt,codec,NULL) < 0){
//         av_log(NULL,AV_LOG_ERROR,"不能打开编码器  \n");
//         return -1;
//     }
//     avcodec_parameters_from_context(stream->codecpar,codec_cxt);

//     //写入文件头
//     avformat_write_header(output_cxt,NULL);
//     int size = codec_cxt->width * codec_cxt->height;

//     AVPacket* packet;
//     av_new_packet(packet,size * 3);

//     int got_picture = 0;
//     int result = avcodec_encode_video2(codec_cxt,packet,frame,&got_picture);
//     if(result < 0){
//         av_log(NULL,AV_LOG_ERROR,"编码失败  \n");
//         return -1;
//     }
//     printf("got_picture %d \n",got_picture);
//     if(got_picture == 1){
//         //将packet中的数据写入本地文件
//         result = av_write_frame(output_cxt,packet);
//     }
//     av_free_packet(packet);
//     //将流尾写入输出媒体文件并释放文件数据
//     av_write_trailer(output_cxt);
//     if(frame){
//         av_frame_unref(frame);
//     }
//     avio_close(output_cxt->pb);
//     avformat_free_context(output_cxt);
//     return 0;
// }



int savePicture(AVFrame* pFrame, int iFrame) {

    char out_name[32];

    // Open file
    sprintf(out_name, "frame%d.jpg", iFrame);


    int width = pFrame->width;
    int height = pFrame->height;

    AVCodecContext *pCodeCtx = NULL;

    
    AVFormatContext *pFormatCtx = avformat_alloc_context();
    // 设置输出文件格式
    pFormatCtx->oformat = av_guess_format("mjpeg", NULL, NULL);

    // 创建并初始化输出AVIOContext
    if (avio_open(&pFormatCtx->pb, out_name, AVIO_FLAG_READ_WRITE) < 0) {
        printf("Couldn't open output file.");
        return -1;
    }

    // 构建一个新stream
    AVStream *pAVStream = avformat_new_stream(pFormatCtx, 0);
    if (pAVStream == NULL) {
        return -1;
    }

    AVCodecParameters *parameters = pAVStream->codecpar;
    parameters->codec_id = pFormatCtx->oformat->video_codec;
    parameters->codec_type = AVMEDIA_TYPE_VIDEO;
    parameters->format = AV_PIX_FMT_YUVJ420P;
    parameters->width = width;
    parameters->height = height;

    AVCodec *pCodec = avcodec_find_encoder(pAVStream->codecpar->codec_id);

    if (!pCodec) {
        printf("Could not find encoder\n");
        return -1;
    }

    pCodeCtx = avcodec_alloc_context3(pCodec);
    if (!pCodeCtx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    if ((avcodec_parameters_to_context(pCodeCtx, pAVStream->codecpar)) < 0) {
        fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return -1;
    }

    pCodeCtx->time_base = (AVRational) {1, 25};

    if (avcodec_open2(pCodeCtx, pCodec, NULL) < 0) {
        printf("Could not open codec.");
        return -1;
    }

    int ret = avformat_write_header(pFormatCtx, NULL);
    if (ret < 0) {
        printf("write_header fail\n");
        return -1;
    }

    int y_size = pCodeCtx->width * pCodeCtx->height;

    //Encode
    // 给AVPacket分配足够大的空间
    AVPacket pkt;
    av_new_packet(&pkt, y_size * 3);

    // 编码数据
    ret = avcodec_send_frame(pCodeCtx, pFrame);
    if (ret < 0) {
        printf("Could not avcodec_send_frame.");
        return -1;
    }

    // 得到编码后数据
    ret = avcodec_receive_packet(pCodeCtx, &pkt);
    if (ret < 0) {
        printf("Could not avcodec_receive_packet");
        return -1;
    }

    ret = av_write_frame(pFormatCtx, &pkt);

    if (ret < 0) {
        printf("Could not av_write_frame");
        return -1;
    }

    av_packet_unref(&pkt);

    //Write Trailer
    av_write_trailer(pFormatCtx);


    avcodec_close(pCodeCtx);
    avio_close(pFormatCtx->pb);
    avformat_free_context(pFormatCtx);

    return 0;
}

void video_codec_init()
{
    int i;
    device_type = AV_HWDEVICE_TYPE_CUDA;
    /* find the video decoder */
    decoder = avcodec_find_decoder_by_name("h264_cuvid");
    if (!decoder) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }


    for (i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
        if (!config) {
            fprintf(stderr, "Decoder %s does not support device type %s.\n",
                    decoder->name, av_hwdevice_get_type_name(device_type));
            return;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == device_type) {
            hw_pix_fmt = config->pix_fmt;
            break;
        }
    }

    decoder_ctx = avcodec_alloc_context3(decoder);
    if (!decoder_ctx) {
        fprintf(stderr, "Could not allocate codec context\n");
        exit(1);
    }

    // First set the hw device then set the hw frame
    decoder_ctx->get_format  = get_hw_format;

    int err = 0;

    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, device_type,
                                      NULL, NULL, 0)) < 0) {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return;
    }
    decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    /* open it */
    if (avcodec_open2(decoder_ctx, decoder, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }
    decoder_ctx->width = 1280;
    decoder_ctx->height = 720;
    printf("video codec initialized.\n");
}


void set_decoder_ctx_params(int w, int h)
{
    decoder_ctx->width = w;
    decoder_ctx->height = h;
}

int i = 0;
int firstTime = 0;
struct SwsContext* swsContext = NULL;

#define RESCALE 1 

void decode_feed(AVCodecContext *dec_ctx, AVPacket *pkt, AVFrame *frame, int timestamp)
{
    // int i, ch;
    int ret, data_size;
    AVFrame *pFrameRGB;
    // pFrameRGB->format = AV_PIX_FMT_RGB24;
    // pFrameRGB->width = 400;
    // pFrameRGB->height = 300;
    AVFrame *swFrame;
    pFrameRGB = av_frame_alloc();
    swFrame = av_frame_alloc();
    if (NULL == pFrameRGB || NULL == swFrame) {
        fprintf(stderr, "Alloc frame failed!\n");
        return;
    }

    uint8_t *buffer = NULL;
    int numBytes = 0;

    // calculate buffer size after decoding and allocate buffer
    #if !RESCALE
    // numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, decoder_ctx->width, decoder_ctx->height, 1);
    numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUVJ420P, decoder_ctx->width, decoder_ctx->height, 1);
    buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
    // av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_RGB24, decoder_ctx->width, decoder_ctx->height, 1);
    av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_YUVJ420P, decoder_ctx->width, decoder_ctx->height, 1);

    #else
    // numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, 400, 300, 1);
    numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUVJ420P, 400, 300, 1);
    buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
    // av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_RGB24, 400, 300, 1);
    av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_YUVJ420P, 400, 300, 1);
    #endif
    /* send the packet with the compressed data to the decoder */
    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error submitting the packet to the decoder\n");
        goto fail;
    }
    if (firstTime == 0) {
        firstTime = 1;
        #if !RESCALE
        swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_NV12, decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_YUVJ420P, SWS_BICUBIC, NULL, NULL, NULL);
        #else
        // swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_NV12, 400, 300, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
        swsContext = sws_getContext(decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_NV12, 400, 300, AV_PIX_FMT_YUVJ420P, SWS_BICUBIC, NULL, NULL, NULL);
        #endif
    }

    /* read all the output frames (in general there may be any number of them */
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        // printf("receive frame ret = %d\n", ret);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_freep(&buffer);
            av_frame_free(&pFrameRGB);
            av_frame_free(&swFrame);
            return;

        }
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            goto fail;
        }

        // download frame from gpu to cpu
        ret = av_hwframe_transfer_data(swFrame, frame, 0);
        if (ret < 0) {
            fprintf(stderr, "Error transferring the data to system memory\n");
            goto fail;
        }



        if (swsContext == NULL) {
            printf("swsContext failed.\n");
            goto fail;
        }
        #if !RESCALE
        sws_scale(swsContext, (const unsigned char* const*)swFrame->data, swFrame->linesize, 0, decoder_ctx->height, pFrameRGB->data, pFrameRGB->linesize);
        // SaveFrame(pFrameRGB, decoder_ctx->width, decoder_ctx->height, timestamp);
        pFrameRGB->format = AV_PIX_FMT_YUVJ420P;
        pFrameRGB->width = 1280;
        pFrameRGB->height = 720;
        savePicture(pFrameRGB, timestamp);
        #else
        sws_scale(swsContext, (const unsigned char* const*)swFrame->data, swFrame->linesize, 0, decoder_ctx->height, pFrameRGB->data, pFrameRGB->linesize);
        
        pFrameRGB->format = AV_PIX_FMT_YUVJ420P;
        pFrameRGB->width = 400;
        pFrameRGB->height = 300;
        savePicture(pFrameRGB, timestamp);
        #endif
    }
fail:
    av_freep(&buffer);
    av_free(&pFrameRGB);
    av_free(&swFrame);
    // sws_freeContext(swsContext);
    return;    
}

void ds_feedpkt(char* pktdata, int pktsize, int timestamp){
    AVFrame *decoded_frame = NULL;  
    AVPacket        packet;
    av_init_packet(&packet);

    packet.data = pktdata;
    packet.size = pktsize;
    // printf("pktsize %d\n", pktsize);
    if (!decoded_frame) {
        if (!(decoded_frame = av_frame_alloc())) {
            fprintf(stderr, "Could not allocate audio frame\n");
            return;
        }
    }   
    decode_feed(decoder_ctx, &packet, decoded_frame, timestamp);
    if (decoded_frame != NULL)
        av_frame_free(&decoded_frame);
    av_packet_unref(&packet);
    return;
}
