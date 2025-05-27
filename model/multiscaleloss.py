import torch
import torch.nn.functional as F


def epe(output_flow, target_flow):
    epe_map = torch.norm(target_flow - output_flow, 2, 1, keepdim=True)
    return epe_map
    

def multi_epe(flow4, flow5, flow6, target_flow, target_mask):
    def scaled_for_epe(output, target_flow, target_mask):
        b, _, h, w = output.size()
        target_mask_scaled = F.interpolate(target_mask, (h, w), mode='bilinear', align_corners=False)
        target_mask_scaledH = (target_mask_scaled > 0.0).float()
        target_flow_scaled = F.interpolate(target_flow, (h, w), mode='bilinear', align_corners=False)
        epe_map = epe(output * target_mask_scaledH, target_flow_scaled)
        return epe_map.sum()

    loss = 0
    loss += scaled_for_epe(flow6, target_flow, target_mask) * 16.0
    loss += scaled_for_epe(flow5, target_flow, target_mask) * 4.0
    epeFull = epe(flow4 * target_mask, target_flow)
    epeFullSum = epeFull.sum()
    loss += epeFullSum
    return loss

    
def bce(maskd, maskr, target_maskd, target_maskr):
    criterion = torch.nn.BCELoss(reduction='sum')
    
    loss = 0
    loss += criterion(maskd, target_maskd) * 0.15
    loss += criterion(maskr, target_maskr) * 0.15
    return loss


def masked_epe(output, target_flow, target_mask):
    b, _, h, w = target_flow.size()
    upsampled_output = output * target_mask
    epe_map = epe(upsampled_output, target_flow)
    valid_pixels = torch.sum(target_mask)
    masked_epe = epe_map.sum() / valid_pixels
    return masked_epe


def mask_bce(maskd, maskr, target_maskd, target_maskr):
    criterion = torch.nn.BCELoss()
    
    loss = 0
    loss += criterion(maskd, target_maskd) * 0.5
    loss += criterion(maskr, target_maskr) * 0.5
    return loss
