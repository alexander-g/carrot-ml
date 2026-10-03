from src.cells_yolo_maskhead import (
    MaskHead, 
    MaskHeadTrainStep, 
    convert_boxmasks_into_full_result
)

import torch



def test_trainstep():
    mask = torch.zeros([1, 512, 512], dtype=torch.int64)
    mask[10:20, 10:20] = 1
    mask[50:70, 70:80] = 2
    raw_batch = [ 
        (
            torch.zeros([3,512,512]),
            mask,
        )
    ]

    head = MaskHead()
    step = MaskHeadTrainStep(head)
    if torch.cuda.is_available():
        step = step.cuda()
    loss, logs = step(raw_batch)



def test_paste_boxmasks():
    shape = (512,512)
    masks = torch.ones([3,64,64])
    # hollow
    masks[2, 5:-5, 5:-5] = 0
    boxes = torch.tensor([
        (10,   20,  40, 40),
        (100, 200, 240,300),
        (400, 400, 440,440),
    ])


    output = convert_boxmasks_into_full_result(shape, boxes, masks)
    assert output.shape == (3,512,512)
    assert output[:,:19].abs().sum() == 0
    assert output[:,:,:9].abs().sum() == 0
    assert output[:,41:199].abs().sum() == 0
    assert output[:,:,41:99].abs().sum() == 0
    assert (output[0,20:40,10:40] > 0).all()
    assert (output[1,200:300,100:240] > 0).all()
    assert output[1:, :50, :50].abs().sum() == 0
    assert output[:,415:425, 415:425].abs().sum() == 0



    # handle negative coordinates without errors
    boxes1 = torch.tensor([
        (-10,   -20,  40, 40),
    ])
    output1 = convert_boxmasks_into_full_result(shape, boxes1, masks[:1])
    assert output1.shape == (1,512,512)
    assert (output1[:,:40,:40] > 0).all()
    assert (output1[:,41:] == 0).all()
    assert (output1[:,:,41:] == 0).all()


    # handle no masks without errors
    output2 = convert_boxmasks_into_full_result(shape, boxes[:0], masks[:0])
    assert output2.shape == (0,512,512)
    assert output2.abs().sum() == 0


