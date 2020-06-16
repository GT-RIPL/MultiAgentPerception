import torch.nn as nn
import torch
import torchvision.models as models

import pretrainedmodels

from ptsemseg.models.utils import conv2DBatchNormRelu, deconv2DBatchNormRelu, Sparsemax
import random



class n_segnet_encoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=3):
        super(n_segnet_encoder, self).__init__()
        self.in_channels = in_channels

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(self.in_channels, 64, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(64              , 64, k_size=3, stride=2, padding=1)

        # down 2 
        self.conv3 = conv2DBatchNormRelu(64              ,128, k_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(128             ,128, k_size=3, stride=2, padding=1)

        # down 3
        self.conv5 = conv2DBatchNormRelu(128             , 256, k_size=3, stride=1, padding=1)
        self.conv6 = conv2DBatchNormRelu(256             , 256, k_size=3, stride=1, padding=1)
        self.conv7 = conv2DBatchNormRelu(256             , 256, k_size=3, stride=2, padding=1)

        # down 4
        self.conv8 = conv2DBatchNormRelu(256             , 512, k_size=3, stride=1, padding=1)
        self.conv9 = conv2DBatchNormRelu(512             , 512, k_size=3, stride=1, padding=1)
        self.conv10= conv2DBatchNormRelu(512             , 512, k_size=3, stride=2, padding=1)

        # down 5
        self.conv11= conv2DBatchNormRelu(512             , 512, k_size=3, stride=1, padding=1)
        self.conv12= conv2DBatchNormRelu(512             , 512, k_size=3, stride=1, padding=1)
        self.conv13= conv2DBatchNormRelu(512             , 512, k_size=3, stride=2, padding=1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)
        outputs = self.conv7(outputs)
        outputs = self.conv8(outputs)
        outputs = self.conv9(outputs)
        outputs = self.conv10(outputs)
        outputs = self.conv11(outputs)
        outputs = self.conv12(outputs)
        outputs = self.conv13(outputs)
        return outputs


class resnet_encoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=3):
        super(resnet_encoder, self).__init__()
        feat_chn = 256
        #self.feature_backbone = n_segnet_encoder(n_classes=n_classes, in_channels=in_channels)
        self.feature_backbone = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None)

        self.backbone_0  = self.feature_backbone.conv1
        self.backbone_1  = nn.Sequential(self.feature_backbone.bn1, self.feature_backbone.relu, self.feature_backbone.maxpool, self.feature_backbone.layer1)
        self.backbone_2  = self.feature_backbone.layer2
        self.backbone_3 = self.feature_backbone.layer3
        self.backbone_4 = self.feature_backbone.layer4


    def forward(self, inputs):
        # print('input:')
        # print(inputs.size())
        # import pdb; pdb.set_trace()
        outputs = self.backbone_0(inputs)
        # print('base_0 size: ')
        # print(base_0.size())

        outputs = self.backbone_1(outputs)
        # print('base_1 size: ')
        # print(base_1.size())

        outputs = self.backbone_2(outputs)
        # print('base_2 size: ')
        # print(base_2.size())

        outputs = self.backbone_3(outputs)
        # print('base_3 size: ')
        # print(base_3.size())

        outputs = self.backbone_4(outputs)
        # print('base_4 size: ')
        # print(base_4.size())

        return outputs

### ============= Decoder Backbone ============= ###
class n_segnet_decoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=512):
    #def __init__(self, n_classes=21, in_channels=512,agent_num=5):
        super(n_segnet_decoder, self).__init__()
        self.in_channels = in_channels
        # Decoder
        self.deconv1= deconv2DBatchNormRelu(self.in_channels, 512, k_size=3, stride=2, padding=1,output_padding=1)
        self.deconv2= conv2DBatchNormRelu(512             , 512, k_size=3, stride=1, padding=1)
        self.deconv3= conv2DBatchNormRelu(512             , 512, k_size=3, stride=1, padding=1)

        # up 4
        self.deconv4= deconv2DBatchNormRelu(512           , 512, k_size=3, stride=2, padding=1,output_padding=1)
        self.deconv5= conv2DBatchNormRelu(512             , 512, k_size=3, stride=1, padding=1)
        self.deconv6= conv2DBatchNormRelu(512             , 256, k_size=3, stride=1, padding=1)

        # up 3
        self.deconv7= deconv2DBatchNormRelu(256           , 256, k_size=3, stride=2, padding=1,output_padding=1)
        self.deconv8= conv2DBatchNormRelu(256             , 128, k_size=3, stride=1, padding=1)

        # up 2
        self.deconv9= deconv2DBatchNormRelu(128           , 128, k_size=3, stride=2, padding=1,output_padding=1)
        self.deconv10= conv2DBatchNormRelu(128             , 64, k_size=3, stride=1, padding=1)

        # up 1
        self.deconv11= deconv2DBatchNormRelu(64            , 64, k_size=3, stride=2, padding=1,output_padding=1)
        self.deconv12= conv2DBatchNormRelu(64              , n_classes, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.deconv1(inputs)
        outputs = self.deconv2(outputs)
        outputs = self.deconv3(outputs)

        outputs = self.deconv4(outputs)
        outputs = self.deconv5(outputs)
        outputs = self.deconv6(outputs)
        outputs = self.deconv7(outputs)
        outputs = self.deconv8(outputs)
        outputs = self.deconv9(outputs)
        outputs = self.deconv10(outputs)
        outputs = self.deconv11(outputs)
        outputs = self.deconv12(outputs)
        return outputs


class simple_decoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=512):
        super(simple_decoder, self).__init__()
        self.in_channels = in_channels

        feat_chn = 256

        self.pred = nn.Sequential(
            nn.Conv2d(self.in_channels, feat_chn, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chn, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        pred = self.pred(inputs)
        # print('pred size: ')
        # print(pred.size())
        pred = nn.functional.interpolate(pred, size=torch.Size([inputs.size()[2]*32,inputs.size()[3]*32]), mode='bilinear', align_corners=False)
        # print('pred size: ')
        #rint(pred.size())

        return pred


class FCN_decoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=512):
        super(FCN_decoder, self).__init__()
        feat_chn = 256

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, feat_chn, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chn, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        pred = self.pred(base_4)
        print('pred size: ')
        print(pred.size())
        pred = nn.functional.interpolate(pred, size=inputs.size()[-2:], mode='bilinear', align_corners=False)
        print('pred size: ')
        print(pred.size())

        return pred

