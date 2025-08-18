

def loop_closure(estimated_c2ws, final=False):
    '''
    Compute loop closure correction
    
    returns: 
        None or the pose correction for each keyframe
    '''
    print("\nDetecting loop closures ...")
    # first see if current submap generates any new edge to the pose graph
    correction_list = []
    c2ws_est = estimated_c2ws.detach()     
    submap_paths = sorted(glob.glob(str(submap_path/"*.ckpt")), key=lambda x: int(x.split('/')[-1][:-5]))
    
    if self.submap_id < 3 or len(self.detect_closure(self.submap_id)) == 0:
        print(f"\nNo loop closure detected at submap no.{self.submap_id}")
        return correction_list
    
    pose_graph, odometry_edges, loop_edges = self.construct_pose_graph(final)
    
    # save pgo edge analysis result
    
    if len(loop_edges) > 0 and len(loop_edges) > self.n_loop_edges:
        
        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=self.config['lc']['pgo_edge_prune_thres'],
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),      # 优化算法: LM
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),     # 收敛条件
            option)
        
        self.pgo_count += 1
        self.n_loop_edges = len(loop_edges)
        
        # 为每个子地图生成校正变换矩阵
        for id in range(self.submap_id+1):
            submap_correction = {
                'submap_id': id,
                "correct_tsfm": pose_graph.nodes[id].pose}
            correction_list.append(submap_correction)
            
        self.analyse_pgo(odometry_edges, loop_edges, pose_graph)
    
    else:
        print("No valid loop edges or new loop edges. Skipping ...")
        
    return correction_list