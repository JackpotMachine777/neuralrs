#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::{attention as gpu_attention, attention_batch as gpu_attention_batch, LSTMCell, RNNCell};
use neuralrs::tensor::Tensor;

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < tol, "{what} [{i}]: gpu {} cpu {}", a[i], b[i]);
    }
}

#[test]
fn cuda_attention() {
    let (seq, d) = (4usize, 6usize);
    let q: Vec<f32> = (0..seq * d).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let k: Vec<f32> = (0..seq * d).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let v: Vec<f32> = (0..seq * d).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let seed: Vec<f32> = (0..seq * d).map(|i| (i % 5) as f32 * 0.1 + 0.2).collect();

    let cq = Node::new(q.clone(), vec![seq, d]);
    let ck = Node::new(k.clone(), vec![seq, d]);
    let cv = Node::new(v.clone(), vec![seq, d]);
    let cout = neuralrs::nn::attention::attention(cq.clone(), ck.clone(), cv.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    let (cdq, cdk, cdv) = (cq.borrow().grad.clone(), ck.borrow().grad.clone(), cv.borrow().grad.clone());

    let gq = Node::new(q, vec![seq, d]);
    let gk = Node::new(k, vec![seq, d]);
    let gv = Node::new(v, vec![seq, d]);
    gpu::to_cuda(&gq); gpu::to_cuda(&gk); gpu::to_cuda(&gv);
    let gout = gpu_attention(&gq, &gk, &gv);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gdq, gdk, gdv) = (gpu::read_grad(&gq), gpu::read_grad(&gk), gpu::read_grad(&gv));

    close(&gf, &cf, 1e-3, "attention fwd");
    close(&gdq, &cdq, 1e-3, "attention dq");
    close(&gdk, &cdk, 1e-3, "attention dk");
    close(&gdv, &cdv, 1e-3, "attention dv");
    println!("attention: gpu matches cpu");
}

#[test]
fn cuda_attention_batch() {
    let (batch, seq, d) = (2usize, 3usize, 4usize);
    let n = batch * seq * d;
    let q: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let k: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let v: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let seed: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.1 + 0.2).collect();

    let cq = Node::new(q.clone(), vec![batch, seq, d]);
    let ck = Node::new(k.clone(), vec![batch, seq, d]);
    let cv = Node::new(v.clone(), vec![batch, seq, d]);
    let cout = neuralrs::nn::attention::attention_batch(cq.clone(), ck.clone(), cv.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    let (cdq, cdk, cdv) = (cq.borrow().grad.clone(), ck.borrow().grad.clone(), cv.borrow().grad.clone());

    let gq = Node::new(q, vec![batch, seq, d]);
    let gk = Node::new(k, vec![batch, seq, d]);
    let gv = Node::new(v, vec![batch, seq, d]);
    gpu::to_cuda(&gq); gpu::to_cuda(&gk); gpu::to_cuda(&gv);
    let gout = gpu_attention_batch(&gq, &gk, &gv);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gdq, gdk, gdv) = (gpu::read_grad(&gq), gpu::read_grad(&gk), gpu::read_grad(&gv));

    close(&gf, &cf, 1e-3, "attention_batch fwd");
    close(&gdq, &cdq, 1e-3, "attention_batch dq");
    close(&gdk, &cdk, 1e-3, "attention_batch dk");
    close(&gdv, &cdv, 1e-3, "attention_batch dv");
    println!("attention_batch: gpu matches cpu");
}

#[test]
fn cuda_rnn_step() {
    let (batch, is, hs) = (3usize, 4usize, 5usize);
    let x: Vec<f32> = (0..batch * is).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let h: Vec<f32> = (0..batch * hs).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let w_xh: Vec<f32> = (0..is * hs).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let w_hh: Vec<f32> = (0..hs * hs).map(|i| (i % 9) as f32 * 0.1 - 0.4).collect();
    let bias: Vec<f32> = (0..hs).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
    let seed: Vec<f32> = (0..batch * hs).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let mut cell = neuralrs::nn::rnn::RNNCell {
        w_xh: Tensor::new(w_xh.clone(), vec![is, hs]),
        w_hh: Tensor::new(w_hh.clone(), vec![hs, hs]),
        bias: Tensor::new(bias.clone(), vec![hs]),
        input_size: is, hidden_size: hs,
        w_xh_node: None, w_hh_node: None, bias_node: None,
    };
    let cx = Node::new(x.clone(), vec![batch, is]);
    let ch = Node::new(h.clone(), vec![batch, hs]);
    let cout = cell.step(cx.clone(), ch.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    cell.sync_grads();
    let (cdxh, cdhh, cdb) = (cell.w_xh.grad.clone(), cell.w_hh.grad.clone(), cell.bias.grad.clone());
    let (cdx, cdh) = (cx.borrow().grad.clone(), ch.borrow().grad.clone());

    let gw_xh = Node::new(w_xh, vec![is, hs]);
    let gw_hh = Node::new(w_hh, vec![hs, hs]);
    let gbias = Node::new(bias, vec![hs]);
    gpu::to_cuda(&gw_xh); gpu::to_cuda(&gw_hh); gpu::to_cuda(&gbias);
    let gcell = RNNCell::new(gw_xh.clone(), gw_hh.clone(), gbias.clone(), is, hs);
    let gx = Node::new(x, vec![batch, is]);
    let gh = Node::new(h, vec![batch, hs]);
    gpu::to_cuda(&gx); gpu::to_cuda(&gh);
    let gout = gcell.step(&gx, &gh);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gdxh, gdhh, gdb) = (gpu::read_grad(&gw_xh), gpu::read_grad(&gw_hh), gpu::read_grad(&gbias));
    let (gdx, gdh) = (gpu::read_grad(&gx), gpu::read_grad(&gh));

    close(&gf, &cf, 1e-3, "rnn fwd");
    close(&gdxh, &cdxh, 1e-3, "rnn dw_xh");
    close(&gdhh, &cdhh, 1e-3, "rnn dw_hh");
    close(&gdb, &cdb, 1e-3, "rnn dbias");
    close(&gdx, &cdx, 1e-3, "rnn dx");
    close(&gdh, &cdh, 1e-3, "rnn dh");
    println!("rnn step: gpu matches cpu");
}

#[test]
fn cuda_lstm_step() {
    let (batch, is, hs) = (2usize, 4usize, 5usize);
    let (wsz, usz, bsz) = (is * hs, hs * hs, hs);
    let mk = |len: usize, s: usize| -> Vec<f32> { (0..len).map(|i| ((i + s) % 13) as f32 * 0.1 - 0.6).collect() };

    let (wf, uf, bf) = (mk(wsz, 1), mk(usz, 2), mk(bsz, 3));
    let (wi, ui, bi) = (mk(wsz, 4), mk(usz, 5), mk(bsz, 6));
    let (wo, uo, bo) = (mk(wsz, 7), mk(usz, 8), mk(bsz, 9));
    let (wg, ug, bg) = (mk(wsz, 10), mk(usz, 11), mk(bsz, 12));
    let x = mk(batch * is, 20); let h = mk(batch * hs, 21); let c = mk(batch * hs, 22);
    let seed = mk(batch * hs, 30);

    let t = |d: &Vec<f32>, sh: Vec<usize>| Tensor::new(d.clone(), sh);
    let mut cell = neuralrs::nn::lstm::LSTMCell {
        w_f: t(&wf, vec![is, hs]), u_f: t(&uf, vec![hs, hs]), b_f: t(&bf, vec![hs]),
        w_i: t(&wi, vec![is, hs]), u_i: t(&ui, vec![hs, hs]), b_i: t(&bi, vec![hs]),
        w_o: t(&wo, vec![is, hs]), u_o: t(&uo, vec![hs, hs]), b_o: t(&bo, vec![hs]),
        w_g: t(&wg, vec![is, hs]), u_g: t(&ug, vec![hs, hs]), b_g: t(&bg, vec![hs]),
        input_size: is, hidden_size: hs, nodes: None,
    };
    let cx = Node::new(x.clone(), vec![batch, is]);
    let ch = Node::new(h.clone(), vec![batch, hs]);
    let cc = Node::new(c.clone(), vec![batch, hs]);
    let (chn, ccn) = cell.step(cx.clone(), ch.clone(), cc.clone());
    chn.borrow_mut().grad = seed.clone();
    backward_graph(&chn);
    let hf = chn.borrow().data.clone();
    let cfwd = ccn.borrow().data.clone();
    cell.sync_grads();
    let cdw = [
        cell.w_f.grad.clone(), cell.u_f.grad.clone(), cell.b_f.grad.clone(),
        cell.w_i.grad.clone(), cell.u_i.grad.clone(), cell.b_i.grad.clone(),
        cell.w_o.grad.clone(), cell.u_o.grad.clone(), cell.b_o.grad.clone(),
        cell.w_g.grad.clone(), cell.u_g.grad.clone(), cell.b_g.grad.clone(),
    ];
    let (cdx, cdh, cdc) = (cx.borrow().grad.clone(), ch.borrow().grad.clone(), cc.borrow().grad.clone());

    let node = |d: &Vec<f32>, sh: Vec<usize>| { let n = Node::new(d.clone(), sh); gpu::to_cuda(&n); n };
    let (gwf, guf, gbf) = (node(&wf, vec![is, hs]), node(&uf, vec![hs, hs]), node(&bf, vec![hs]));
    let (gwi, gui, gbi) = (node(&wi, vec![is, hs]), node(&ui, vec![hs, hs]), node(&bi, vec![hs]));
    let (gwo, guo, gbo) = (node(&wo, vec![is, hs]), node(&uo, vec![hs, hs]), node(&bo, vec![hs]));
    let (gwg, gug, gbg) = (node(&wg, vec![is, hs]), node(&ug, vec![hs, hs]), node(&bg, vec![hs]));
    let gcell = LSTMCell {
        w_f: gwf.clone(), u_f: guf.clone(), b_f: gbf.clone(),
        w_i: gwi.clone(), u_i: gui.clone(), b_i: gbi.clone(),
        w_o: gwo.clone(), u_o: guo.clone(), b_o: gbo.clone(),
        w_g: gwg.clone(), u_g: gug.clone(), b_g: gbg.clone(),
        input_size: is, hidden_size: hs,
    };
    let gx = node(&x, vec![batch, is]);
    let gh = node(&h, vec![batch, hs]);
    let gc = node(&c, vec![batch, hs]);
    let (ghn, gcn) = gcell.step(&gx, &gh, &gc);
    let ghf = gpu::to_host(&ghn);
    let gcf = gpu::to_host(&gcn);
    gpu::set_grad(&ghn, &seed);
    backward_graph(&ghn);
    let gdw = [
        gpu::read_grad(&gwf), gpu::read_grad(&guf), gpu::read_grad(&gbf),
        gpu::read_grad(&gwi), gpu::read_grad(&gui), gpu::read_grad(&gbi),
        gpu::read_grad(&gwo), gpu::read_grad(&guo), gpu::read_grad(&gbo),
        gpu::read_grad(&gwg), gpu::read_grad(&gug), gpu::read_grad(&gbg),
    ];
    let (gdx, gdh, gdc) = (gpu::read_grad(&gx), gpu::read_grad(&gh), gpu::read_grad(&gc));

    close(&ghf, &hf, 1e-3, "lstm h_new fwd");
    close(&gcf, &cfwd, 1e-3, "lstm c_new fwd");
    let names = ["w_f", "u_f", "b_f", "w_i", "u_i", "b_i", "w_o", "u_o", "b_o", "w_g", "u_g", "b_g"];
    for j in 0..12 { close(&gdw[j], &cdw[j], 1e-3, &format!("lstm d{}", names[j])); }
    close(&gdx, &cdx, 1e-3, "lstm dx");
    close(&gdh, &cdh, 1e-3, "lstm dh_prev");
    close(&gdc, &cdc, 1e-3, "lstm dc_prev");
    println!("lstm step: gpu matches cpu");
}