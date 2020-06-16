import copy
import torchvision.models as models


from ptsemseg.models.agent import Single_agent, All_agents, LearnWho2Com, LearnWhen2Com, MIMOcom,MIMO_All_agents, MIMOcomWho


def get_model(model_dict, n_classes, version=None):
    name = model_dict["model"]["arch"]

    model = _get_model_instance(name)
    in_channels = 3
    if name == "Single_agent":
        model = model(n_classes=n_classes, in_channels=in_channels,
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone'],
                      feat_squeezer=model_dict["model"]['feat_squeezer'],
                      feat_channel=model_dict["model"]['feat_channel'])
    elif name == "All_agents":        
          model = model(n_classes=n_classes, in_channels=in_channels, 
                             aux_agent_num=model_dict["model"]['agent_num'], 
                             shuffle_flag=model_dict["model"]['shuffle_features'],
                             enc_backbone=model_dict["model"]['enc_backbone'],
                             dec_backbone=model_dict["model"]['dec_backbone'],
                             feat_squeezer=model_dict["model"]['feat_squeezer'],
                             feat_channel=model_dict["model"]['feat_channel'])
    elif name == "MIMO_All_agents":
          model = model(n_classes=n_classes, in_channels=in_channels,
                        aux_agent_num=model_dict["model"]['agent_num'],
                        shuffle_flag=model_dict["model"]['shuffle_features'],
                        enc_backbone=model_dict["model"]['enc_backbone'],
                        dec_backbone=model_dict["model"]['dec_backbone'],
                        feat_squeezer=model_dict["model"]['feat_squeezer'],
                        feat_channel=model_dict["model"]['feat_channel'])

    elif name == "LearnWho2Com":
        model = model(n_classes=n_classes, in_channels=in_channels, 
                      attention=model_dict["model"]['attention'],has_query=model_dict["model"]['query'],
                      sparse=model_dict["model"]['sparse'],
                      aux_agent_num=model_dict["model"]['agent_num'],
                      shared_img_encoder=model_dict["model"]["shared_img_encoder"],
                      image_size=model_dict["data"]["img_rows"],
                      query_size=model_dict["model"]["query_size"],key_size=model_dict["model"]["key_size"],
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone']
                      )
    elif name == "LearnWhen2Com":
        model = model(n_classes=n_classes, in_channels=in_channels,
                      attention=model_dict["model"]['attention'],has_query=model_dict["model"]['query'],
                      sparse=model_dict["model"]['sparse'],
                      aux_agent_num=model_dict["model"]['agent_num'],
                      shared_img_encoder=model_dict["model"]["shared_img_encoder"],
                      image_size=model_dict["data"]["img_rows"],
                      query_size=model_dict["model"]["query_size"],key_size=model_dict["model"]["key_size"],
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone']
                      )
    elif name == "MIMOcom":
        model = model(n_classes=n_classes, in_channels=in_channels,
                      attention=model_dict["model"]['attention'],has_query=model_dict["model"]['query'],
                      sparse=model_dict["model"]['sparse'],
                      agent_num=model_dict["model"]['agent_num'],
                      shared_img_encoder=model_dict["model"]["shared_img_encoder"],
                      image_size=model_dict["data"]["img_rows"],
                      query_size=model_dict["model"]["query_size"],key_size=model_dict["model"]["key_size"],
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone']
                      )
    elif name == "MIMOcomWho":
        model = model(n_classes=n_classes, in_channels=in_channels,
                      attention=model_dict["model"]['attention'],has_query=model_dict["model"]['query'],
                      sparse=model_dict["model"]['sparse'],
                      agent_num=model_dict["model"]['agent_num'],
                      shared_img_encoder=model_dict["model"]["shared_img_encoder"],
                      image_size=model_dict["data"]["img_rows"],
                      query_size=model_dict["model"]["query_size"],key_size=model_dict["model"]["key_size"],
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone']
                      )

    else:
        model = model(n_classes=n_classes, in_channels=in_channels,
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone'])

    return model


def _get_model_instance(name):
    try:
        return {
            "Single_agent": Single_agent,
            "All_agents": All_agents,
            "MIMO_All_agents": MIMO_All_agents,
            'LearnWho2Com':LearnWho2Com,
            'LearnWhen2Com': LearnWhen2Com,
            'MIMOcom': MIMOcom,
            'MIMOcomWho': MIMOcomWho,
        }[name]
    except:
        raise ("Model {} not available".format(name))
