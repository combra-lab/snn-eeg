#include <stdlib.h>
#include "nxsdk.h"
#include "bias_encoding.h"
#include "bias_encoding_info.h"

int do_encoder(runState *s){
    return 1;
}

void run_encoder(runState *s){
    int time = s->time_step;

    // Inject spikes to bias neurons
    for(int ii=0; ii<BIAS_DIMENSION; ii++){
        if(time % WINDOW_STEP >= bias_start_list[ii] && time % WINDOW_STEP < bias_end_list[ii]){
            uint16_t axonId = 1<<14 | ((bias_axon_id_list[ii]) & 0x3FFF);
            ChipId chipId = nx_nth_chipid(BIAS_CHIP_ID);
            CoreId coreId = nx_nth_coreid(bias_core_id_list[ii]);
            nx_send_remote_event(time, chipId, coreId, axonId);
        }
    }
}