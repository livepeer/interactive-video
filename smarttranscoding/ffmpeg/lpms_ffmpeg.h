#ifndef _LPMS_DEEPSPEECH_H_
#define _LPMS_DEEPSPEECH_H_
#include <libavutil/hwcontext.h>
#include <libavutil/rational.h>
#include <libswresample/swresample.h>
#ifndef MAX_AUDIO_FRAME_SIZE
#define MAX_AUDIO_FRAME_SIZE 32000
#endif

#ifndef MAX_AUDIO_BUFFER_SIZE
#define MAX_AUDIO_BUFFER_SIZE 1024000
#endif


#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

typedef struct Audioinfo{
    int input_channels;
    int input_rate;
    int input_nb_samples;
    enum AVSampleFormat input_sample_fmt;
} Audioinfo;

typedef struct {
	char*  buffer;
	size_t buffer_size;
} ds_audio_buffer;


#define MAX_STACK_SIZE 0x1000000
#define PROCESSOR_UNIT 640

typedef struct {
    char pbuffer[MAX_STACK_SIZE];
    int nPos;
} STACK;

typedef struct {
    int initialized;
} struct_transcribe_thread;

typedef struct {
    // SwrContext* resample_ctx;
    const AVCodec *codec;
    AVCodecContext *c;
} codec_params;

// int deepspeech_init();

// void audio_codec_init();
// void audio_codec_deinit();
void video_codec_init();
void ds_feedpkt(char* pktdata, int pktsize, int timestamp);

void set_decoder_ctx_params(int w, int h);
#endif