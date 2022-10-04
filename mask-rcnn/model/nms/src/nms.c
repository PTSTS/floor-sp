#include <TH/TH.h>
#include <math.h>

int cpu_nms(THLongTensor * keep_out, THLongTensor * num_out, THFloatTensor * boxes, THLongTensor * order, THFloatTensor * areas, float nms_overlap_thresh) {
    // boxes has to be sorted
    THArgCheck(THLongTensor_isContiguous(keep_out), 0, "keep_out must be contiguous");
    THArgCheck(THLongTensor_isContiguous(boxes), 2, "boxes must be contiguous");
    THArgCheck(THLongTensor_isContiguous(order), 3, "order must be contiguous");
    THArgCheck(THLongTensor_isContiguous(areas), 4, "areas must be contiguous");
    // Number of ROIs
    long boxes_num = THFloatTensor_size(boxes, 0);
    long boxes_dim = THFloatTensor_size(boxes, 1);

    long * keep_out_flat = THLongTensor_data(keep_out);
    float * boxes_flat = THFloatTensor_data(boxes);
    long * order_flat = THLongTensor_data(order);
    float * areas_flat = THFloatTensor_data(areas);

    THByteTensor* suppressed = THByteTensor_newWithSize1d(boxes_num);
    THByteTensor_fill(suppressed, 0);
    unsigned char * suppressed_flat =  THByteTensor_data(suppressed);

    // nominal indices
    int i, j;
    // sorted indices
    int _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;

    long num_to_keep = 0;
    int boxes_length = sizeof keep_out_flat / sizeof *keep_out_flat;
    
    int keep_num = THFloatTensor_size(keep_out, 0);
    for (int i = 0; i < keep_num; i ++) {
        keep_out_flat[i] = 0;
    }

    printf("boxes_length %d,", boxes_length);
    printf("boxes_num %d,", boxes_num);
    for (_i=0; _i < boxes_num; _i++) {
        i = order_flat[_i];
        if (suppressed_flat[i] == 1) {
            keep_out_flat[num_to_keep] = 0;
            continue;
        }
        keep_out_flat[num_to_keep++] = i;
        printf("%d,", i);
        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iarea = areas_flat[i];
        for (_j = _i + 1; _j < boxes_num; _j++) {
            j = order_flat[_j];
            if (suppressed_flat[j] == 1) {
                continue;
            }
            xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + 1);
            h = fmaxf(0.0, yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (iarea + areas_flat[j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_flat[j] = 1;
            }
        }
    }

    float* keep_out_new = THLongTensor_data(keep_out);
    keep_out_new[1] = 0.0;
    printf("\nsupressed_flat");
    for (int i = 0; i < keep_num; i ++) {
        printf(" %d,", suppressed_flat[i]);
    }

    printf("\nkeep_out_flat");
    for (int i = 0; i < keep_num; i ++) {
        if (i%2 - 1 == 0) {
            keep_out_flat[i] = 0;
        }
        printf(" %d,", keep_out_flat[i]);
    }
    printf("\n order flat");
    for (int i = 0; i < keep_num; i ++) {
        printf(" %d,", order_flat[i]);
    }
    printf("\n num_out_new");

    long* num_out_new = THLongTensor_data(num_out);
    for (int i = 0; i < boxes_num; i ++) {
        printf(" %d,", num_out_new[i]);
    }
    printf("\n");

    long *num_out_flat = THLongTensor_data(num_out);
    *num_out_flat = num_to_keep;
    printf("num_to_keep %d\n", num_to_keep);
    printf("num_out_flat %d\n", *num_out_flat);
    THByteTensor_free(suppressed);
    return 1;
}