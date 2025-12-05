def freeze_layers(model, mode):
    if mode == "none":
        # 全層学習
        for p in model.parameters():
            p.requires_grad = True
        return model.model.parameters()

    elif mode == "fc_only":
        # 全層凍結 → FCだけ学習
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.fc.parameters():
            p.requires_grad = True
        return model.model.fc.parameters()
    
    elif mode == "layer4_fc":
        #全層凍結 → 4層以降を学習
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.layer4.parameters():
            p.requires_grad = True
        for p in model.model.fc.parameters():
            p.requires_grad = True
        return list(model.model.layer4.parameters()) + list(model.model.fc.parameters())
    
    elif mode == "deep_blocks":
        #全層凍結 → 3層以降を学習
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.layer3.parameters():
            p.requires_grad = True
        for p in model.model.layer4.parameters():
            p.requires_grad = True
        for p in model.model.fc.parameters():
            p.requires_grad = True
        return list(model.model.layer3.parameters()) \
            + list(model.model.layer4.parameters()) \
            + list(model.model.fc.parameters())
    
    elif mode == "freeze_only_layer":
        #全層凍結 → conv1とFC層を学習
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'layer' not in name:
                param.requires_grad = True
        return list(filter(lambda p: p.requires_grad, model.parameters()))