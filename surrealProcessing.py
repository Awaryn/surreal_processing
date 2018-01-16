import numpy as np
import os.path
import scipy.io
import h5py

from tqdm import tqdm

id_root=0
joints=24

res_x = 320  # *scn.render.resolution_x
res_y = 240  # *scn.render.resolution_y

min_zind = 1
max_zind = 64

def getIntrinsicBlender():
    # These are set in Blender (datageneration/main_part1.py)
    f_mm             = 60  # *cam_ob.data.lens
    sensor_w_mm      = 32  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y / res_x # *cam_ob.data.sensor_height (function of others)

    scale = 1; # *scn.render.resolution_percentage/100
    skew  = 0; # only use rectangular pixels
    pixel_aspect_ratio = 1;

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x * scale / sensor_w_mm;
    fy_px = f_mm * res_y * scale * pixel_aspect_ratio / sensor_h_mm;

    # Center of the image
    u = res_x * scale / 2.0;
    v = res_y * scale / 2.0;

    # Intrinsic camera matrix
    K = np.array(
        [[fx_px, skew,  u],
         [0,     fy_px, v],
         [0,     0,     1]]
    )

    return K


def getExtrinsicBlender(T):
#   returns extrinsic camera matrix
#
#   T : translation vector from Blender (*cam_ob.location)
#   RT: extrinsic computer vision camera matrix
#   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

# Take the first 3 columns of the matrix_world in Blender and transpose.
# This is hard-coded since all images in SURREAL use the same.

    R_world2bcam = np.array(
        [[ 0,  0, -1],
         [ 0, -1,  0],
         [ 1,  0,  0]]
    )

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = np.dot(R_world2bcam, -np.squeeze(T))

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array(
        [[1,  0,  0],
         [0, -1,  0],
         [0,  0, -1]]
    )

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    # Put into 3x4 matrix
    RT = np.c_[R_world2cv, T_world2cv]

    return RT

def surreal2pavlakos(path, output):

    L = np.loadtxt(os.path.join(path, "list.txt"), dtype=str)

    centers_list = []
    zinds_list   = []
    scales_list  = []
    parts_list   = []
    video_ids_list = []
    frame_ids_list = []
    videos_list = []

    total_used_frames = 0
    total_frames = 0
    for i, f in enumerate(tqdm(L)):
        M = scipy.io.loadmat(os.path.join(path, f + "_info.mat"))

        J2D = M['joints2D'].reshape(2, joints, -1).transpose(2, 1, 0)
        J3D = M['joints3D'].reshape(3, joints, -1).transpose(2, 1, 0)

        sequence = M['sequence'][0]
        camLoc = M['camLoc']

        frames = J2D.shape[0]

        K = getIntrinsicBlender()
        RT = getExtrinsicBlender(camLoc)

        V_ones = np.ones((*J3D.shape[:2], 1))
        V = np.concatenate((J3D, V_ones), axis=-1)

        # Calculate the coordinate of the joint in camera space and
        # Substract the root joint (as explicited in Pavlakos et al)
        camV = np.sum(RT[None, None, :, :] * V[:, :, None, :], axis=-1)
        camV = camV - camV[:, [id_root], :]

        zind = (0.5 * camV[:, :, 2] + 0.5) * (max_zind - min_zind) + min_zind
        zind = np.round(np.clip(zind, min_zind, max_zind, out=zind))

        max_J2D = np.max(J2D, axis=1)
        min_J2D = np.min(J2D, axis=1)
#         centers = 0.5 * (max_J2D + min_J2D)       # Join bounding box center
        centers = J2D[:,0,:]                        # Take the root as center
        dist = np.min(np.array([[res_x, res_y]]) / (max_J2D - min_J2D), axis=-1)
        scales = 1.5 / dist                         # 1.5 is a magical constant here

        # Compute frames in which the human is entirely in the screen
        saved_frames = np.all(np.logical_and(min_J2D >= np.array([[0, 0]]),
                                             max_J2D <= np.array([[res_x, res_y]])),
                              axis=1)

        saved_ids = np.arange(frames)[saved_frames]
        video_ids = [i] * len(saved_ids)

#         if (len(saved_ids) != frames):
#             print(f, len(saved_ids), frames)

        video_ids_list.extend(video_ids)
        frame_ids_list.extend(saved_ids)

        centers_list.extend(centers[saved_ids])
        scales_list.extend(scales[saved_ids])
        zinds_list.extend(zind[saved_ids])
        parts_list.extend(J2D[saved_ids])

        videos_list.append(f + '.mp4')

        total_used_frames += len(saved_ids)
        total_frames += frames

    print("used frames : %d/%d [%.2f%%]" % (total_used_frames, total_frames, 100 * total_used_frames / total_frames))


    with h5py.File(output + '.h5', 'w') as f:

        f.create_dataset("center", data=np.array(centers_list, dtype='f8'))
        f.create_dataset("part", data=np.array(parts_list, dtype='f8'))
        f.create_dataset("scale", data=np.array(scales_list, dtype='f8'))
        f.create_dataset("zind", data=np.array(zinds_list, dtype='u1'))
        f.create_dataset("video_id", data=np.array(video_ids_list, dtype='u4'))
        f.create_dataset("frame_id", data=np.array(frame_ids_list, dtype='u4'))


    np.savetxt((output + '_videos.txt'), np.array(videos_list), fmt='%s')


#reduce the dataset size
def reduceAnnotation(input, compression, output):
    with h5py.File(input, 'r') as i:

        N = len(i['zind'])
        M = int(compression * N)
        R = np.random.choice(N, M)
        R.sort()

        with h5py.File(output, 'w') as o:
            o.create_dataset("center", data=np.array(i["center"])[R])
            o.create_dataset("part", data=np.array(i["part"])[R])
            o.create_dataset("scale", data=np.array(i["scale"])[R])
            o.create_dataset("zind", data=np.array(i["zind"])[R])
            o.create_dataset("video_id", data=np.array(i["video_id"])[R])
            o.create_dataset("frame_id", data=np.array(i["frame_id"])[R])
