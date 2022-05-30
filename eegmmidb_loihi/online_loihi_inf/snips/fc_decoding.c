#include <stdlib.h>
#include "nxsdk.h"
#include "fc_decoding.h"
#include "fc_decoding_info.h"

int output_weighted_fc_activity[FC_DIMENSION] = {0};

int do_decoder(runState *s){
    return 1;
}

void run_decoder(runState *s){
    int time = s->time_step;

    // Compute weighted output FC layer activity
    int dt = 0;
    if(time % WINDOW_STEP >= DECODE_START_STEP && (time % WINDOW_STEP <= DECODE_END_STEP || time % WINDOW_STEP == 0)){
        if(time % WINDOW_STEP != 0){
            dt = time % WINDOW_STEP - DECODE_START_STEP;
        }
        else{
            dt = WINDOW_STEP - DECODE_START_STEP;
        }
        for(int ii=0; ii<FC_DIMENSION; ii++){
            if(SPIKE_COUNT[(time)&3][ii+0x20] > 0){
                output_weighted_fc_activity[ii] += ts_weight[dt];
            }
            SPIKE_COUNT[(time)&3][ii+0x20] = 0;
        }
    }

    // Write weighted output FC layer activity to host
    if(time % WINDOW_STEP == DECODE_END_STEP){
        int output_channel_id = getChannelID("decodeoutput");
        writeChannel(output_channel_id, output_weighted_fc_activity, FC_DIMENSION);
        for(int ii=0; ii<FC_DIMENSION; ii++){
            output_weighted_fc_activity[ii] = 0;
        }
    }
}