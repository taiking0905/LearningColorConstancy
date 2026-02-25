def freeze_layers(model, mode):
    trainable_params = []

    # ---------- 全層学習 ----------
    if mode == "none":
        for p in model.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        return trainable_params

    # ---------- FCのみ学習 ----------
    elif mode == "fc_only":
        for p in model.parameters():
            p.requires_grad = False

        for p in model.fc.parameters():
            p.requires_grad = True
            trainable_params.append(p)

        return trainable_params

    # ---------- layer4 + FC ----------
    elif mode == "layer4_fc":
        for p in model.parameters():
            p.requires_grad = False

        for name, p in model.backbone.named_parameters():
            if "layer4" in name:
                p.requires_grad = True
                trainable_params.append(p)

        for p in model.fc.parameters():
            p.requires_grad = True
            trainable_params.append(p)

        return trainable_params

    # ---------- layer3, layer4 + FC ----------
    elif mode == "deep_blocks":
        for p in model.parameters():
            p.requires_grad = False

        for name, p in model.backbone.named_parameters():
            if "layer3" in name or "layer4" in name:
                p.requires_grad = True
                trainable_params.append(p)

        for p in model.fc.parameters():
            p.requires_grad = True
            trainable_params.append(p)

        return trainable_params

    # ---------- conv1 + FC ----------
    elif mode == "freeze_only_layer":
        for p in model.parameters():
            p.requires_grad = False

        for name, p in model.backbone.named_parameters():
            if name.startswith("0"):  # conv1 は backbone[0]
                p.requires_grad = True
                trainable_params.append(p)

        for p in model.fc.parameters():
            p.requires_grad = True
            trainable_params.append(p)

        return trainable_params

    else:
        raise ValueError(f"Unknown FREEZE_MODE: {mode}")
