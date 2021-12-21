from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("alignment/step1/forward_torch.npy")
    paddle_info = diff_helper.load_info("alignment/step1/forward_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="alignment/step1/forward_diff.log")