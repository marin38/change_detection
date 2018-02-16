def pad_int_zeros(i, num_digits):
    res = str(i)
    for j in range(num_digits-len(res)):
        res = '0'+res
    return res


def split_image(dataset_dir, img_name, model, size = (192,240), shear = 0.25, dataset_type='AICD'):
    invert_gt = None
    
    if dataset_type == 'TSUNAMI':
        invert_gt = True
        img_1_path = dataset_dir+'t0/'+img_name+'.jpg'
        img_2_path = dataset_dir+'t1/'+img_name+'.jpg'
        img_gt_path = dataset_dir+'ground_truth/'+img_name+'.bmp'
        
    elif dataset_type == 'AICD':
        invert_gt = False
        img_1_path = dataset_dir+'Images_NoShadow/'+img_name+'_moving.png'
        img_2_path = dataset_dir+'Images_NoShadow/'+img_name+'_target.png'
        img_gt_path = dataset_dir+'GroundTruth/'+img_name+'_gtmask.png'
        
    else:
        raise ValueError('dataset_type must be one of [\'AICD\', \'TSUNAMI\']')

        
    img_1 = cv2.imread(img_1_path)
    img_2 = cv2.imread(img_2_path)
    img_gt = cv2.imread(img_gt_path,0)
    assert img_1 is not None
    assert img_2 is not None
    assert img_gt is not None

    if invert_gt:
        img_gt = cv2.bitwise_not(img_gt)
        
    img = np.empty((img_1.shape[0], img_1.shape[1], 6))
    img[:, :, :3] = img_1/255
    img[:, :, 3:] = img_2/255
    #img_gt = img_gt/255
                   
    h,w,_ = img.shape
    # Create reflective padding
    shear_int = (int(size[0]*shear),int(size[1]*shear))
    pad_h_1 = (size[0]-shear_int[0]-(h-size[0])%(size[0]-shear_int[0]))//2
    pad_h_2 = (size[0]-shear_int[0]-(h-size[0])%(size[0]-shear_int[0]))//2+(size[0]-shear_int[0]-(h-size[0])%(size[0]-shear_int[0]))%2
    pad_w_1 = (size[1]-shear_int[1]-(w-size[1])%(size[1]-shear_int[1]))//2
    pad_w_2 = (size[1]-shear_int[1]-(w-size[1])%(size[1]-shear_int[1]))//2+(size[1]-shear_int[1]-(w-size[1])%(size[1]-shear_int[1]))%2
    #print(pad_h_1,pad_h_2,pad_w_1,pad_w_2)
    img = np.pad(img,((pad_h_1,pad_h_2),(pad_w_1,pad_w_2),(0,0)), 'reflect')
    
    # Split image into patches
    h,w,_ = img.shape
    img_arr = []
    n_rows = (h-size[0])//(size[0]-shear_int[0])+1
    n_cols = (w-size[1])//(size[1]-shear_int[1])+1
    for i in range(n_rows):
        for j in range(n_cols):
            coord_h = i*(size[0]-shear_int[0])
            coord_w = j*(size[1]-shear_int[1])
            img_crop = img[coord_h:coord_h+size[0], coord_w:coord_w+size[1],:]
            img_arr.append(img_crop)
 
    # Create predictions for patches
    mask_arr = []
    for img_crop in img_arr:
        img_to_pred = np.array([img_crop])
        mask_pred = model.predict(img_to_pred)[0]
        mask_arr.append(mask_pred)
    
    # Merge masks in one
    index = 0
    channels = 1
    final_pred = np.zeros((h,w))
    for i in range (n_rows):
        for j in range (n_cols):
            #print(i,j)
            coord_h = i*(size[0]-shear_int[0])
            coord_w = j*(size[1]-shear_int[1])
            
            shift_h = shear_int[0]//2
            shift_w = shear_int[1]//2
            
            mask_to_append = mask_arr[index]
            mask_to_append[0:shift_h,:]=0
            mask_to_append[size[0]-shift_h:,:]=0
            mask_to_append[:,0:shift_w]=0
            mask_to_append[:,size[1]-shift_w:]=0
                
            final_pred[coord_h:coord_h+size[0], coord_w:coord_w+size[1]]+=mask_to_append
            index+=1
    final_pred = final_pred[pad_h_1:h-pad_h_2, pad_w_1:w-pad_w_2]
    return final_pred, img_gt