

def process_video(ctx: VideoProcessingContext):
    with tqdm(total=ctx.video_info.total_frames) as pbar:
        for frame in ctx.frame_generator:
            # all your logic, using ctx.model, ctx.model_params, ctx.output_dir, etc.
            ...