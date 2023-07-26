def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    用于训练的方法
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    time1=time.time()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach() # target, for second to end.e.g."好吗？<EOS>"
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device, dtype=torch.long) # input. e.g. "how are you?"
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        # 每100步打印日志
        if _ % 100 == 0 and _!=0:
            time2=time.time()
            print(_,"epoch:"+str(epoch)+"-loss:"+str(loss)+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))
            # training_logger.add_row(str(epoch), str(_), str(loss))
            # console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
