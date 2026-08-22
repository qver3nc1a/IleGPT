# using gpt to generate text

import torch
from IleGPT import IleGPT

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model = IleGPT(GPT_CONFIG_124M)
model.eval()


import tiktoken


# from IleGPT import generate_text_simple
# generating text
def generate_text_simple(
    model, idx, max_new_tokens, context_size
):  # idx is a (batch, n_tokens) array of indices
    # print(idx)
    # print(idx.shape)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # crop content if exceeds context size
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)  # shape (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # shape (batch, 1)
        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # appends sampled idx to running sequence with idx shape (batch, n_tokens+1)
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # unsqueeze adds the batch dimension


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze()  # removes batch dimension
    return tokenizer.decode(flat.tolist())


# utility function to implement cross entropy loss of given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# function to compute loss over all batches
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # iterate over all batches if num not specified
    else:
        num_batches = min(
            num_batches, len(data_loader)
        )  # reduces num_batches to be max num_batches
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # sums loss for each batch
        else:
            break
    return total_loss / num_batches  # averages loss over batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # dropout is disabled for stable results
    with torch.no_grad():  # to reduce computational overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # compact print format
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = (
        [],
        [],
        [],
    )  # lists to track losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # main training loop
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # calculate loss gradients
            optimizer.step()  # upgrade model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

        if global_step % eval_freq == 0:  # optional evaluation step
            train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(
                f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
            )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )  # print a sample text after each epoch
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    # simple example (untrained model produces gibberish)
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    # print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # two inputs
    inputs = torch.tensor(
        [[16833, 3626, 6100], [40, 1107, 588]]
    )  # every effort moves you, i really like
    targets = torch.tensor(
        [[3626, 6100, 345], [1107, 588, 11311]]
    )  # effort moves you forward, really like chocolate

    # feed inputs to model -> calculate logits -> apply softmax -> transform logits to probas
    with torch.no_grad():  # not training so no gradient tracking yet
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    # print(probas.shape)  # torch.Size([2, 3, 50257])

    # obtain highest probas and tokenIDs
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("Token IDs:\n", token_ids)

    # convert tokenIDs back to text
    # print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    # print(f"Outputs batch 1:" f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # evaluate performance via a loss
    # initial softmax probability scores
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # print("Text 2:", target_probas_2)

    # log probas
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    # print(log_probas)

    # combine log probas into a single score
    avg_log_probas = torch.mean(log_probas)
    # print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    # print(neg_avg_log_probas)

    # flatten logit and target tensors
    # print("Logits shape:", logits.shape)
    # print("Targets shape:", targets.shape)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    # print("Flattened logits:", logits_flat.shape)
    # print("Flattened targets:", targets_flat.shape)

    # cross-entropy takes care of the steps above
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    # print(loss)

    # perplexity
    perp = torch.exp(loss)
    # print("Perplexity of loss: ", perp)

    # apply loss computation to entire train and validation
    # load The Verdict
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # number of characters and tokens
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    # print("Characters:", total_characters)
    # print("Tokens:", total_tokens)

    # divide into training and validation sets
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # create data loader
    from Dataloader import create_dataloader_v1

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # check dataloaders
    # print("Train loader:")
    # for x, y in train_loader:
    # print(x.shape, y.shape)
    # print("\nValidation loader:")
    # for x, y in val_loader:
    # print(x.shape, y.shape)

    # apply loss function to training and validation batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    # print("Training loss: ", train_loss)
    # print("Validation loss: ", val_loss)

torch.manual_seed(123)
model = IleGPT(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")
num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)
