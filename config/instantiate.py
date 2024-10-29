from detectron2.config import instantiate


def instantiate_cmdise(cfg):
    backbone = instantiate(cfg.backbone)
    # cfg.sem_seg_head.input_shape = backbone.output_shape()
    # cfg.sem_seg_head.pixel_decoder.input_shape = backbone.output_shape()
    cfg.backbone = backbone
    model = instantiate(cfg)

    return model
