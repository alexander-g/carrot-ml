from src.cells_yolo_maskhead import MaskHead, MaskHeadTrainStep
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

    print('done')



