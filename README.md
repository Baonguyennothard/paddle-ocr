# paddle-ocr

`python predict_rec.py --image_dir="23893m.png" --rec_model_dir="en_PP-OCRv3_rec_srn" --rec_image_shape="1,64,256" --rec_char_dict_path="en_PP-OCRv3_rec_srn\sidecode_dict.txt" --rec_algorithm="SRN" --use_gpu=False`

[-h] [--use_gpu USE_GPU] [--use_xpu USE_XPU] [--use_npu USE_NPU] [--ir_optim IR_OPTIM]
                      [--use_tensorrt USE_TENSORRT] [--min_subgraph_size MIN_SUBGRAPH_SIZE] [--precision PRECISION]
                      [--gpu_mem GPU_MEM] [--gpu_id GPU_ID] [--image_dir IMAGE_DIR] [--page_num PAGE_NUM]
                      [--det_algorithm DET_ALGORITHM] [--det_model_dir DET_MODEL_DIR] [--det_limit_side_len DET_LIMIT_SIDE_LEN]       
                      [--det_limit_type DET_LIMIT_TYPE] [--det_box_type DET_BOX_TYPE] [--det_db_thresh DET_DB_THRESH]
                      [--det_db_box_thresh DET_DB_BOX_THRESH] [--det_db_unclip_ratio DET_DB_UNCLIP_RATIO]
                      [--max_batch_size MAX_BATCH_SIZE] [--use_dilation USE_DILATION] [--det_db_score_mode DET_DB_SCORE_MODE]
                      [--det_east_score_thresh DET_EAST_SCORE_THRESH] [--det_east_cover_thresh DET_EAST_COVER_THRESH]
                      [--det_east_nms_thresh DET_EAST_NMS_THRESH] [--det_sast_score_thresh DET_SAST_SCORE_THRESH]
                      [--det_sast_nms_thresh DET_SAST_NMS_THRESH] [--det_pse_thresh DET_PSE_THRESH]
                      [--det_pse_box_thresh DET_PSE_BOX_THRESH] [--det_pse_min_area DET_PSE_MIN_AREA]
                      [--det_pse_scale DET_PSE_SCALE] [--scales SCALES] [--alpha ALPHA] [--beta BETA]
                      [--fourier_degree FOURIER_DEGREE] [--rec_algorithm REC_ALGORITHM] [--rec_model_dir REC_MODEL_DIR]
                      [--rec_image_inverse REC_IMAGE_INVERSE] [--rec_image_shape REC_IMAGE_SHAPE] [--rec_batch_num REC_BATCH_NUM]     
                      [--max_text_length MAX_TEXT_LENGTH] [--rec_char_dict_path REC_CHAR_DICT_PATH]
                      [--use_space_char USE_SPACE_CHAR] [--vis_font_path VIS_FONT_PATH] [--drop_score DROP_SCORE]
                      [--e2e_algorithm E2E_ALGORITHM] [--e2e_model_dir E2E_MODEL_DIR] [--e2e_limit_side_len E2E_LIMIT_SIDE_LEN]       
                      [--e2e_limit_type E2E_LIMIT_TYPE] [--e2e_pgnet_score_thresh E2E_PGNET_SCORE_THRESH]
                      [--e2e_char_dict_path E2E_CHAR_DICT_PATH] [--e2e_pgnet_valid_set E2E_PGNET_VALID_SET]
                      [--e2e_pgnet_mode E2E_PGNET_MODE] [--use_angle_cls USE_ANGLE_CLS] [--cls_model_dir CLS_MODEL_DIR]
                      [--cls_image_shape CLS_IMAGE_SHAPE] [--label_list LABEL_LIST] [--cls_batch_num CLS_BATCH_NUM]
                      [--cls_thresh CLS_THRESH] [--enable_mkldnn ENABLE_MKLDNN] [--cpu_threads CPU_THREADS]
                      [--use_pdserving USE_PDSERVING] [--warmup WARMUP] [--sr_model_dir SR_MODEL_DIR]
                      [--sr_image_shape SR_IMAGE_SHAPE] [--sr_batch_num SR_BATCH_NUM] [--draw_img_save_dir DRAW_IMG_SAVE_DIR]
                      [--save_crop_res SAVE_CROP_RES] [--crop_res_save_dir CROP_RES_SAVE_DIR] [--use_mp USE_MP]
                      [--total_process_num TOTAL_PROCESS_NUM] [--process_id PROCESS_ID] [--benchmark BENCHMARK]
                      [--save_log_path SAVE_LOG_PATH] [--show_log SHOW_LOG] [--use_onnx USE_ONNX]


Namespace(alpha=1.0, benchmark=False, beta=1.0, cls_batch_num=6, cls_image_shape='3, 48, 192', cls_model_dir=None, cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det_algorithm='DB', det_box_type='quad', det_db_box_thresh=0.6, det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, det_limit_side_len=960, det_limit_type='max', det_model_dir=None, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, det_pse_thresh=0, det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results', drop_score=0.5, e2e_algorithm='PGNet', e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast', e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5, gpu_id=0, gpu_mem=500, image_dir='32907m.png', ir_optim=True, label_list=['0', '180'], max_batch_size=10, max_text_length=25, min_subgraph_size=15, page_num=0, precision='fp32', process_id=0, rec_algorithm='SRN', rec_batch_num=6, rec_char_dict_path='en_PP-OCRv3_rec_srn\\sidecode_dict.txt', rec_image_inverse=True, rec_image_shape='1,64,256', rec_model_dir='en_PP-OCRv3_rec_srn', save_crop_res=False, save_log_path='./log_output/', scales=[8, 16, 32], show_log=True, sr_batch_num=1, sr_image_shape='3, 32, 128', sr_model_dir=None, total_process_num=1, use_angle_cls=False, use_dilation=False, use_gpu=False, use_mp=False, use_npu=False, use_onnx=False, use_pdserving=False, use_space_char=True, use_tensorrt=False, use_xpu=False, vis_font_path='./doc/fonts/simfang.ttf', warmup=False)