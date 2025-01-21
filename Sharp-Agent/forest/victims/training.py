"""Repeatable code parts concerning optimization and training schedules."""

import copy
from collections import defaultdict

import higher
import torch
import torch.nn.functional as F

import wandb

from ..consts import BENCHMARK, NON_BLOCKING
from .batched_attacks import _gradient_matching, construct_attack
from .utils import print_and_save_stats

torch.backends.cudnn.benchmark = BENCHMARK


def check_cosine_similarity(kettle, model, criterion, inputs, labels, step_size):
    device = kettle.setup["device"]
    model.eval()

    intended_labels = torch.tensor([data[1] for data in kettle.source_trainset]).to(
        device=device, dtype=torch.long
    )

    target_images = torch.stack([data[0] for data in kettle.source_trainset]).to(
        **kettle.setup
    )

    outputs_normal = model(inputs)
    try:
        fx, _ = criterion(outputs_normal, labels)
    except:
        fx = criterion(outputs_normal, labels)

    # (B) grads_normal を取得
    grads_normal = torch.autograd.grad(fx, model.parameters(), retain_graph=True)
    grads_normal_flat = torch.cat([g.view(-1) for g in grads_normal])

    # (C) ターゲットバッチの forward
    outputs_target = model(target_images)
    try:
        fx_target, _ = criterion(outputs_target, intended_labels)
    except:
        fx_target = criterion(outputs_target, intended_labels)

    # (D) grads_target を取得
    grads_target = torch.autograd.grad(fx_target, model.parameters())
    grads_target_flat = torch.cat([g.view(-1) for g in grads_target])

    # (E) Cosine Similarity を一回で計算
    cos_sim = F.cosine_similarity(grads_normal_flat, grads_target_flat, dim=0)
    if kettle.args.wandb:
        wandb.log(
            {
                "train_loss": fx.item(),
                "target_loss": fx_target.item(),
                "cosine_similarity": cos_sim.item(),
                "step-size": step_size,
            }
        )
    return cos_sim.item()


def renewal_wolfecondition_stepsize(
    kettle, args, model, loss_fn, alpha, source_trainset, setup
):
    c2, c1 = args.wolfe

    intended_labels = torch.tensor([data[1] for data in source_trainset]).to(
        device=setup["device"], dtype=torch.long
    )

    target_images = torch.stack([data[0] for data in source_trainset]).to(**setup)

    dataset = torch.utils.data.TensorDataset(target_images, intended_labels)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.wolfe_batch, shuffle=True
    )
    copy_model = copy.deepcopy(model)

    fx_total = 0.0
    nabla_fx_total = None
    # for batch_images, batch_labels in dataloader:
    batch_images, batch_labels = dataloader
    fx = loss_fn(copy_model(batch_images), batch_labels)  # 損失を計算

    nabla_fx = torch.autograd.grad(fx, copy_model.parameters(), create_graph=False)

    fx_total += fx.item()
    if nabla_fx_total is None:
        nabla_fx_total = [g.clone() for g in nabla_fx]
    else:
        for i in range(len(nabla_fx_total)):
            nabla_fx_total[i] += nabla_fx[i]

    del copy_model  # 不要な変数を削除してメモリを解放
    torch.cuda.empty_cache()  # メモリを解放

    fx_total /= len(dataloader)
    for i in range(len(nabla_fx_total)):
        nabla_fx_total[i] /= len(dataloader)

    # Wolfe条件を満たす学習率を探索
    max_iters = 40  # 最大で40回の反復を行う
    omega = 0.75  # 学習率の縮小係数
    wolfe_satisfied = False

    def compute_total_loss_and_grad(_model):
        _fx = 0.0
        _grads_accum = None
        for b_images, b_labels in dataloader:
            l = loss_fn(_model(b_images), b_labels)
            _fx += l.item()
            g_ = torch.autograd.grad(l, _model.parameters(), create_graph=False)
            if _grads_accum is None:
                _grads_accum = [gg.clone() for gg in g_]
            else:
                for i in range(len(_grads_accum)):
                    _grads_accum[i] += g_[i]
        _fx /= len(dataloader)
        for i in range(len(_grads_accum)):
            _grads_accum[i] /= float(len(dataloader))
        return _fx, _grads_accum

    # We'll interpret fx_total, nabla_fx_total as the "old" loss and gradient.
    old_fx = fx_total
    old_grad = nabla_fx_total

    # We need the dot(grad, step) for the sufficient-decrease condition
    # The "direction" is typically -grad, so dot(grad, direction) = -||grad||^2
    # but user code attempts to do something like dot(nabla_fx_total[i], updated_params) ...
    # We'll keep a minimal fix and rely on the existing logic.

    # Precompute grad dot grad (we'll need it for the curvature condition)
    old_grad_normdot = sum([torch.sum(g_i * g_i).item() for g_i in old_grad])

    for _ in range(max_iters):
        # -------------------------------------------------
        # (a) Create a temp copy of the original model and do a step: p -= alpha*g
        # -------------------------------------------------
        copy_model_temp = copy.deepcopy(model)
        with torch.no_grad():
            for p, g in zip(copy_model_temp.parameters(), old_grad):
                p.sub_(alpha * g)  # p.data -= alpha*g

        # -------------------------------------------------
        # (b) Compute new loss
        # -------------------------------------------------
        fx_new_total = 0.0
        for batch_images, batch_labels in dataloader:
            fx_new = loss_fn(copy_model_temp(batch_images), batch_labels)
            fx_new_total += fx_new.item()
        fx_new_total /= len(dataloader)

        # -------------------------------------------------
        # (c) Check "sufficient decrease" (Armijo) condition:
        #     fx_new <= fx - c1 * alpha * dot(grad, direction)
        # By default direction = -grad, so dot(grad, direction) = -||grad||^2
        # The user code tries to do sum(torch.dot(g.view(-1), p.view(-1))).
        # We'll replicate it closely but in a safer way:
        # NOTE: A simpler approach:
        #   direction_dot_grad = sum( (g*g).sum() ) (with a minus sign if needed)
        #   but let's preserve the original logic as much as possible.
        # -------------------------------------------------
        direction_dot_grad = 0.0
        # The direction is "delta_p = -alpha*g" from original p to new p
        # But the user code does: sum(torch.dot(g.view(-1), p.view(-1))).
        # That is not typical for standard line-search, but let's keep it.
        with torch.no_grad():
            copy_model_params = list(copy_model_temp.parameters())
            for g_i, p_i in zip(old_grad, copy_model_params):
                direction_dot_grad += torch.dot(g_i.view(-1), p_i.view(-1)).item()

        # The "old_fx - c1 * alpha * <g,d>" part
        # If direction is -g, <g,d> = -||g||^2. But user wants the dot with new p's ...
        # We'll preserve their formula:
        lhs = fx_new_total
        rhs = old_fx - c1 * alpha * direction_dot_grad

        sufficient_decrease = lhs <= rhs

        if sufficient_decrease:
            # -------------------------------------------------
            # (d) Check curvature condition:
            #     dot(nabla_fx_new, old_grad) >= c2 * dot(old_grad, old_grad)
            # -------------------------------------------------
            # We need the new gradient at the new model
            fx_temp, nabla_fx_new_total = compute_total_loss_and_grad(copy_model_temp)

            curvature_condition = True
            for i in range(len(nabla_fx_new_total)):
                lhs_curv = torch.dot(
                    nabla_fx_new_total[i].view(-1), old_grad[i].view(-1)
                )
                rhs_curv = c2 * torch.dot(old_grad[i].view(-1), old_grad[i].view(-1))
                if lhs_curv.item() < rhs_curv.item():
                    curvature_condition = False
                    break

            if curvature_condition:
                wolfe_satisfied = True
                # Done with line search
                del copy_model_temp
                torch.cuda.empty_cache()
                break

        # If we get here, Wolfe not satisfied -> reduce alpha
        alpha *= omega

        # Clean up each iteration
        del copy_model_temp
        torch.cuda.empty_cache()

    if not wolfe_satisfied:
        print("NO WOLFE CONDITION SATISFIED. USING ALPHA=0.01")
        alpha = 0.01

    return alpha


def run_step(
    kettle,
    poison_delta,
    epoch,
    stats,
    model,
    defs,
    optimizer,
    scheduler,
    loss_fn,
    pretraining_phase=False,
):

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    cos_sim, ave_cos = 0, 0
    if pretraining_phase:
        train_loader = kettle.pretrainloader
        valid_loader = kettle.validloader
    else:
        if kettle.args.ablation < 1.0:
            # run ablation on a subset of the training set
            train_loader = kettle.partialloader
        else:
            train_loader = kettle.trainloader
        valid_loader = kettle.validloader
    current_lr = optimizer.param_groups[0]["lr"]
    if "adversarial-cycler" in defs.novel_defense["type"]:
        attackers = []
        for attack in ["wb", "fc", "patch", "htbd", "watermark"]:
            novel_defense = dict(
                type=f"adversarial-{attack}", strength=defs.novel_defense["strength"]
            )
            attackers.append(
                construct_attack(
                    novel_defense,
                    model,
                    loss_fn,
                    kettle.dm,
                    kettle.ds,
                    tau=kettle.args.tau,
                    init="randn",
                    optim="signAdam",
                    num_classes=len(kettle.trainset.classes),
                    setup=kettle.setup,
                )
            )
    elif "adversarial" in defs.novel_defense["type"]:
        attacker = construct_attack(
            defs.novel_defense,
            model,
            loss_fn,
            kettle.dm,
            kettle.ds,
            tau=kettle.args.tau,
            init="randn",
            optim="signAdam",
            num_classes=len(kettle.trainset.classes),
            setup=kettle.setup,
        )

    # Compute flag to activate defenses:
    # Here we are writing these conditions out explicitely:
    if poison_delta is None:  # this is the case if the training set is clean
        if defs.adaptive_attack:
            activate_defenses = True
        else:
            activate_defenses = False
    else:  # this is a poisoned training set
        if defs.defend_features_only:
            activate_defenses = False
        else:
            activate_defenses = True

    for batch, (inputs, labels, ids) in enumerate(train_loader):
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(
            dtype=torch.long, device=kettle.setup["device"], non_blocking=NON_BLOCKING
        )

        # #### Add poison pattern to data #### #
        if poison_delta is not None:
            poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            if len(batch_positions) > 0:
                inputs[batch_positions] += poison_delta[poison_slices].to(
                    **kettle.setup
                )

        # Add data augmentation
        if (
            defs.augmentations
        ):  # defs.augmentations is actually a string, but it is False if --noaugment
            inputs = kettle.augment(inputs)

        # #### Run defenses based on modifying input data #### #
        if activate_defenses:
            if defs.mixing_method["type"] != "":
                inputs, extra_labels, mixing_lmb = kettle.mixer(
                    inputs, labels, epoch=epoch
                )

            # Split Data
            if any(
                [
                    s in defs.novel_defense["type"]
                    for s in ["adversarial", "meta", "combine"]
                ]
            ):
                [temp_sources, inputs, temp_true_labels, labels, temp_fake_label] = (
                    _split_data(
                        inputs,
                        labels,
                        source_selection=defs.novel_defense["source_selection"],
                    )
                )
            # Poison given data ('adversarial patch' for patch attacks lol)
            if "adversarial" in defs.novel_defense["type"]:
                model.eval()
                if "adversarial-cycler" in defs.novel_defense["type"]:
                    attacker = attackers[torch.randint(0, len(attackers), (1,))]
                delta, additional_info = attacker.attack(
                    inputs,
                    labels,
                    temp_sources,
                    temp_true_labels,
                    temp_fake_label,
                    steps=defs.novel_defense["steps"],
                )

                # temp sources are modified for trigger attacks:
                # this already happens as a side effect for hidden-trigger, but not for patch
                if "patch" in defs.novel_defense["type"]:
                    temp_sources = temp_sources + additional_info

                inputs = inputs + delta  # Kind of a reparametrization trick

                if "folded" in defs.novel_defense["type"]:
                    # Fold the input modification and repeat it to both inputs and sources
                    # We discussed two variants of this, folding the updated data and folding the original data
                    if "folded-clean" in defs.novel_defense["type"]:
                        new_inputs = inputs - delta
                    elif "folded-dirty" in defs.novel_defense["type"]:
                        new_inputs = inputs
                    else:
                        raise ValueError(f"Invalid folding option given.")
                    delta, additional_info = attacker.attack(
                        temp_sources,
                        temp_true_labels,
                        new_inputs,
                        labels,
                        temp_fake_label,
                        steps=defs.novel_defense["steps"],
                    )
                    # temp inputs are modified for the folded trigger attacks:
                    if "patch" in defs.novel_defense["type"]:
                        inputs = inputs + additional_info
                    # Modify the sources as well
                    temp_sources = temp_sources + delta

        # Switch into training mode
        list(model.children())[-1].train() if model.frozen else model.train()

        # Change loss function to include corrective terms if mixing with correction
        if (
            defs.mixing_method["type"] != "" and defs.mixing_method["correction"]
        ) and activate_defenses:

            def criterion(outputs, labels):
                return kettle.mixer.corrected_loss(
                    outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn
                )

        else:

            def criterion(outputs, labels):
                loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs.data, dim=1)
                correct_preds = (predictions == labels).sum().item()
                return loss, correct_preds

        # #### Run defenses modifying the loss function #### #
        if activate_defenses:
            # Compute loss
            if "meta" in defs.novel_defense["type"]:
                # Do model updates looking forward for one step
                with higher.innerloop_ctx(
                    model, optimizer, copy_initial_weights=False
                ) as (fmodel, fopt):
                    poison_loss, _ = criterion(fmodel(inputs), labels)
                    fopt.step(poison_loss)
                    outputs = fmodel(temp_sources)
                # Propagate buffers from the functional model to the persistent model:
                # This part is super crucial, otherwise the buffers never get updated!!
                higher.patch.buffer_sync(fmodel, model)

                loss, preds = criterion(outputs, temp_true_labels)
                correct_preds += preds

                if "duplex" in defs.novel_defense["type"]:
                    loss += poison_loss

            elif "lastlayer" in defs.novel_defense["type"]:
                transfer_optimizer = torch.optim.Adam(
                    list(model.children())[-1].parameters(), lr=0.001
                )
                # Do model updates looking forward on the last layer for several steps
                with higher.innerloop_ctx(
                    model, transfer_optimizer, copy_initial_weights=False
                ) as (fmodel, fopt):
                    for idx in range(10):
                        poison_loss, _ = criterion(fmodel(inputs), labels)
                        if idx == 0:
                            # Propagate buffers from the functional model to the persistent model:
                            # This part is super crucial, otherwise the buffers never get updated!!
                            higher.patch.buffer_sync(fmodel, model)
                        fopt.step(poison_loss)
                    outputs = fmodel(temp_sources)

                loss, preds = criterion(outputs, temp_true_labels)
                correct_preds += preds

                if "duplex" in defs.novel_defense["type"]:
                    loss += poison_loss

            elif "recombine" in defs.novel_defense["type"]:
                # Recombine poisoned inputs and sources into a single batch
                inputs = torch.cat((inputs, temp_sources))
                labels = torch.cat((labels, temp_true_labels))

                # Do normal model updates, possibly on modified inputs
                outputs = model(inputs)
                loss, preds = criterion(outputs, labels)
                correct_preds += preds
            else:
                # Do normal model updates, possibly on modified inputs
                outputs = model(inputs)
                loss, preds = criterion(outputs, labels)
                correct_preds += preds
        else:
            # Do normal model updates, possibly on modified inputs
            outputs = model(inputs)
            loss, preds = criterion(outputs, labels)
            correct_preds += preds

        total_preds += labels.shape[0]
        differentiable_params = [p for p in model.parameters() if p.requires_grad]
        # Modify loss with alignment
        if activate_defenses:
            if defs.novel_defense["type"] != "":
                model.eval()
                if defs.novel_defense["type"] == "maximize-alignment-1":
                    temp_labels = torch.randint_like(
                        labels, len(kettle.trainset.classes)
                    )
                    duplicates = temp_labels == labels
                    replacements = temp_labels[duplicates] + torch.randint_like(
                        labels[duplicates], 1, len(kettle.trainset.classes)
                    )
                    temp_labels[duplicates] = replacements % len(
                        kettle.trainset.classes
                    )

                    outputs = model(inputs)
                    poison_grad = torch.autograd.grad(
                        loss_fn(outputs, labels),
                        differentiable_params,
                        create_graph=True,
                    )
                    source_grad = torch.autograd.grad(
                        loss_fn(outputs, temp_labels),
                        differentiable_params,
                        create_graph=True,
                    )
                    loss += defs.novel_defense["strength"] * _gradient_matching(
                        poison_grad, source_grad
                    )
                elif defs.novel_defense["type"] == "maximize-alignment-2":
                    batch_size = inputs.shape[0]
                    shuffle = torch.randperm(batch_size, device=kettle.setup["device"])
                    temp_sources = inputs[shuffle].detach().clone()

                    poison_grad = torch.autograd.grad(
                        loss_fn(model(inputs), labels),
                        differentiable_params,
                        create_graph=True,
                    )
                    source_grad = torch.autograd.grad(
                        loss_fn(model(temp_sources), labels),
                        differentiable_params,
                        create_graph=True,
                    )
                    loss += defs.novel_defense["strength"] * _gradient_matching(
                        poison_grad, source_grad
                    )
                elif defs.novel_defense["type"] == "maximize-source-loss":
                    temp_labels = torch.randint_like(
                        labels, len(kettle.trainset.classes)
                    )
                    duplicates = temp_labels == labels
                    replacements = temp_labels[duplicates] + torch.randint_like(
                        labels[duplicates], 1, len(kettle.trainset.classes)
                    )
                    temp_labels[duplicates] = replacements % len(
                        kettle.trainset.classes
                    )
                    loss -= defs.novel_defense["strength"] * loss_fn(
                        outputs, temp_labels
                    )

        loss.backward()
        epoch_loss += loss.item()

        if activate_defenses:
            with torch.no_grad():
                # Enforce batch-wise privacy if necessary
                # This is a defense discussed in Hong et al., 2020
                # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
                # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
                # of noise to the gradient signal
                if defs.privacy["clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        differentiable_params, defs.privacy["clip"]
                    )
                if defs.privacy["noise"] is not None:
                    loc = torch.as_tensor(0.0, device=kettle.setup["device"])
                    clip_factor = (
                        defs.privacy["clip"]
                        if defs.privacy["clip"] is not None
                        else 1.0
                    )
                    scale = torch.as_tensor(
                        clip_factor * defs.privacy["noise"],
                        device=kettle.setup["device"],
                    )
                    if defs.privacy["distribution"] == "gaussian":
                        generator = torch.distributions.normal.Normal(
                            loc=loc, scale=scale
                        )
                    elif defs.privacy["distribution"] == "laplacian":
                        generator = torch.distributions.laplace.Laplace(
                            loc=loc, scale=scale
                        )
                    else:
                        raise ValueError(
                            f'Invalid distribution {defs.privacy["distribution"]} given.'
                        )
                    for param in differentiable_params:
                        param.grad += generator.sample(param.shape)
        if (
            (epoch >= kettle.args.linesearch_epoch)
            and kettle.args.wolfe
            and poison_delta is not None
        ):
            alpha = renewal_wolfecondition_stepsize(
                kettle,
                kettle.args,
                model,
                loss_fn,
                current_lr * 2,
                kettle.source_trainset,
                kettle.setup,
            )
            optimizer.param_groups[0]["lr"] = alpha
            current_lr = alpha

        optimizer.step()

        if defs.scheduler == "cyclic":
            scheduler.step()
        if kettle.args.dryrun:
            break
    if defs.scheduler == "linear":
        scheduler.step()

    if kettle.args.wandb:
        ave_cos += check_cosine_similarity(
            kettle, model, criterion, inputs, labels, current_lr
        )
    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        predictions, valid_loss = run_validation(
            model,
            loss_fn,
            valid_loader,
            kettle.poison_setup["target_class"],
            kettle.poison_setup["source_class"],
            kettle.setup,
            kettle.args.dryrun,
        )
        source_acc, source_loss, source_clean_acc, source_clean_loss = check_sources(
            model,
            loss_fn,
            kettle.sourceset,
            kettle.poison_setup["target_class"],
            kettle.poison_setup["source_class"],
            kettle.setup,
        )
        # mody
        (
            source_train_acc,
            source_train_loss,
            source_train_clean_acc,
            source_train_clean_loss,
        ) = check_sources(
            model,
            loss_fn,
            kettle.source_trainset,
            kettle.poison_setup["target_train_class"],
            kettle.poison_setup["source_class"],
            kettle.setup,
        )
        # mody
        print(
            f"Source train adv. loss is {source_train_loss:7.4f}, train fool  acc: {source_train_acc:7.2%} | "
            f"train Source orig. loss is {source_train_clean_loss:7.4f}, train orig. acc: {source_train_clean_acc:7.2%} | "
        )
    else:
        predictions, valid_loss = None, None
        source_acc, source_loss, source_clean_acc, source_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]["lr"]
    print_and_save_stats(
        kettle,
        epoch,
        stats,
        current_lr,
        epoch_loss / (batch + 1),
        correct_preds / total_preds,
        predictions,
        valid_loss,
        source_acc,
        source_loss,
        source_clean_acc,
        source_clean_loss,
        cos_sim,
    )


def run_validation(
    model, criterion, dataloader, target_class, source_class, setup, dryrun=False
):
    """Get accuracy of model relative to dataloader.

    Hint: The validation numbers in "target" and "source" explicitely reference the first label in target_class and
    the first label in source_class."""
    model.eval()
    target_class = torch.tensor(target_class).to(
        device=setup["device"], dtype=torch.long
    )
    source_class = torch.tensor(source_class).to(
        device=setup["device"], dtype=torch.long
    )
    predictions = defaultdict(lambda: dict(correct=0, total=0))

    loss = 0

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            labels = labels.to(
                device=setup["device"], dtype=torch.long, non_blocking=NON_BLOCKING
            )
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            predictions["all"]["total"] += labels.shape[0]
            predictions["all"]["correct"] += (predicted == labels).sum().item()

            predictions["target"]["total"] += (labels == target_class[0]).sum().item()
            predictions["target"]["correct"] += (
                (predicted == labels)[labels == target_class[0]].sum().item()
            )

            predictions["source"]["total"] += (labels == source_class).sum().item()
            predictions["source"]["correct"] += (
                (predicted == labels)[labels == source_class].sum().item()
            )

            if dryrun:
                break

    for key in predictions.keys():
        if predictions[key]["total"] > 0:
            predictions[key]["avg"] = (
                predictions[key]["correct"] / predictions[key]["total"]
            )
        else:
            predictions[key]["avg"] = float("nan")

    loss_avg = loss / (i + 1)
    return predictions, loss_avg


def check_sources(model, criterion, sourceset, target_class, original_class, setup):
    """Get accuracy and loss for all sources on their target class."""
    model.eval()
    if len(sourceset) > 0:
        source_images = torch.stack([data[0] for data in sourceset]).to(**setup)
        target_labels = torch.tensor(target_class).to(
            device=setup["device"], dtype=torch.long
        )
        original_labels = torch.stack(
            [
                torch.as_tensor(data[1], device=setup["device"], dtype=torch.long)
                for data in sourceset
            ]
        )
        with torch.no_grad():
            outputs = model(source_images)
            predictions = torch.argmax(outputs, dim=1)

            loss_target = criterion(outputs, target_labels)
            accuracy_target = (
                predictions == target_labels
            ).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            predictions_clean = torch.argmax(outputs, dim=1)
            accuracy_clean = (
                predictions == original_labels
            ).sum().float() / predictions.size(0)

            # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, target: {target_class}')

        return (
            accuracy_target.item(),
            loss_target.item(),
            accuracy_clean.item(),
            loss_clean.item(),
        )
    else:
        return 0, 0, 0, 0


def _split_data(inputs, labels, source_selection="sep-half"):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    #  shuffle/sep-half/sep-1/sep-10
    if source_selection == "shuffle":
        shuffle = torch.randperm(batch_size, device=inputs.device)
        temp_sources = inputs[shuffle].detach().clone()
        temp_true_labels = labels[shuffle].clone()
        temp_fake_label = labels
    elif source_selection == "sep-half":
        temp_sources, inputs = inputs[: batch_size // 2], inputs[batch_size // 2 :]
        temp_true_labels, labels = labels[: batch_size // 2], labels[batch_size // 2 :]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size // 2)
    elif source_selection == "sep-1":
        temp_sources, inputs = inputs[0:1], inputs[1:]
        temp_true_labels, labels = labels[0:1], labels[1:]
        temp_fake_label = labels.mode(keepdim=True)[0]
    elif source_selection == "sep-10":
        temp_sources, inputs = inputs[0:10], inputs[10:]
        temp_true_labels, labels = labels[0:10], labels[10:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(10)
    elif "sep-p" in source_selection:
        p = int(source_selection.split("sep-p")[1])
        p_actual = int(p * batch_size / 128)
        if p_actual > batch_size or p_actual < 1:
            raise ValueError(
                f"Invalid sep-p option given with p={p}. Should be p in [1, 128], "
                f"which will be scaled to the current batch size."
            )
        (
            inputs,
            temp_sources,
        ) = (
            inputs[0:p_actual],
            inputs[p_actual:],
        )
        labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)

    else:
        raise ValueError(f"Invalid selection strategy {source_selection}.")
    return temp_sources, inputs, temp_true_labels, labels, temp_fake_label


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if defs.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            optimized_parameters,
            lr=defs.lr,
            momentum=0.9,
            weight_decay=defs.weight_decay,
            nesterov=True,
        )
    elif defs.optimizer == "SGD-basic":
        optimizer = torch.optim.SGD(
            optimized_parameters,
            lr=defs.lr,
            momentum=0.0,
            weight_decay=defs.weight_decay,
            nesterov=False,
        )
    elif defs.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay
        )
    elif defs.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay
        )

    if defs.scheduler == "cyclic":
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(
            f"Optimization will run over {effective_batches} effective batches in a 1-cycle policy."
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=defs.lr / 100,
            max_lr=defs.lr,
            step_size_up=effective_batches // 2,
            cycle_momentum=True if defs.optimizer in ["SGD"] else False,
        )
    elif defs.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[defs.epochs // 2.667, defs.epochs // 1.6, defs.epochs // 1.142],
            gamma=0.1,
        )
    elif defs.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10_000, 15_000, 25_000], gamma=1
        )

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler
