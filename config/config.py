class OCRConfig:
    def __init__(self, alpha=1.0, benchmark=False, beta=1.0, cls_batch_num=6, cls_image_shape='3, 48, 192', cls_model_dir=None,
                 cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det_algorithm='DB', det_box_type='quad',
                 det_db_box_thresh=0.6, det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5,
                 det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, det_limit_side_len=960,
                 det_limit_type='max', det_model_dir=None, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1,
                 det_pse_thresh=0, det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results',
                 drop_score=0.5, e2e_algorithm='PGNet', e2e_char_dict_path='./ppocr/utils/ic15_dict.txt',
                 e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast',
                 e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5,
                 gpu_id=0, gpu_mem=500, image_dir='32907m.png', ir_optim=True, label_list=['0', '180'],
                 max_batch_size=10, max_text_length=25, min_subgraph_size=15, page_num=0, precision='fp32',
                 process_id=0, rec_algorithm='SRN', rec_batch_num=6, rec_char_dict_path='model/sidecode_dict.txt',
                 rec_image_inverse=True, rec_image_shape='1,64,256', rec_model_dir='model',
                 save_crop_res=False, save_log_path='./log_output/', scales=[8, 16, 32], show_log=True, sr_batch_num=1,
                 sr_image_shape='3, 32, 128', sr_model_dir=None, total_process_num=1, use_angle_cls=False,
                 use_dilation=False, use_gpu=False, use_mp=False, use_npu=False, use_onnx=False, use_pdserving=False,
                 use_space_char=True, use_tensorrt=False, use_xpu=False, vis_font_path='./doc/fonts/simfang.ttf', warmup=False):
        self.alpha = alpha
        self.benchmark = benchmark
        self.beta = beta
        self.cls_batch_num = cls_batch_num
        self.cls_image_shape = cls_image_shape
        self.cls_model_dir = cls_model_dir
        self.cls_thresh = cls_thresh
        self.cpu_threads = cpu_threads
        self.crop_res_save_dir = crop_res_save_dir
        self.det_algorithm = det_algorithm
        self.det_box_type = det_box_type
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_score_mode = det_db_score_mode
        self.det_db_thresh = det_db_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_east_cover_thresh = det_east_cover_thresh
        self.det_east_nms_thresh = det_east_nms_thresh
        self.det_east_score_thresh = det_east_score_thresh
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.det_model_dir = det_model_dir
        self.det_pse_box_thresh = det_pse_box_thresh
        self.det_pse_min_area = det_pse_min_area
        self.det_pse_scale = det_pse_scale
        self.det_pse_thresh = det_pse_thresh
        self.det_sast_nms_thresh = det_sast_nms_thresh
        self.det_sast_score_thresh = det_sast_score_thresh
        self.draw_img_save_dir = draw_img_save_dir
        self.drop_score = drop_score
        self.e2e_algorithm = e2e_algorithm
        self.e2e_char_dict_path = e2e_char_dict_path
        self.e2e_limit_side_len = e2e_limit_side_len
        self.e2e_limit_type = e2e_limit_type
        self.e2e_model_dir = e2e_model_dir
        self.e2e_pgnet_mode = e2e_pgnet_mode
        self.e2e_pgnet_score_thresh = e2e_pgnet_score_thresh
        self.e2e_pgnet_valid_set = e2e_pgnet_valid_set
        self.enable_mkldnn = enable_mkldnn
        self.fourier_degree = fourier_degree
        self.gpu_id = gpu_id
        self.gpu_mem = gpu_mem
        self.image_dir = image_dir
        self.ir_optim = ir_optim
        self.label_list = label_list
        self.max_batch_size = max_batch_size
        self.max_text_length = max_text_length
        self.min_subgraph_size = min_subgraph_size
        self.page_num = page_num
        self.precision = precision
        self.process_id = process_id
        self.rec_algorithm = rec_algorithm
        self.rec_batch_num = rec_batch_num
        self.rec_char_dict_path = rec_char_dict_path
        self.rec_image_inverse = rec_image_inverse
        self.rec_image_shape = rec_image_shape
        self.rec_model_dir = rec_model_dir
        self.save_crop_res = save_crop_res
        self.save_log_path = save_log_path
        self.scales = scales
        self.show_log = show_log
        self.sr_batch_num = sr_batch_num
        self.sr_image_shape = sr_image_shape
        self.sr_model_dir = sr_model_dir
        self.total_process_num = total_process_num
        self.use_angle_cls = use_angle_cls
        self.use_dilation = use_dilation
        self.use_gpu = use_gpu
        self.use_mp = use_mp
        self.use_npu = use_npu
        self.use_onnx = use_onnx
        self.use_pdserving = use_pdserving
        self.use_space_char = use_space_char
        self.use_tensorrt = use_tensorrt
        self.use_xpu = use_xpu
        self.vis_font_path = vis_font_path
        self.warmup = warmup

