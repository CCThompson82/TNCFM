"""This is the session call associated with AUTOENCODER_GRAPH.py"""

logs_path = os.getcwd()+'/TB_logs/'+version_ID

with tf.Session(graph = graph) as session :
    if restore_model == False :
        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        batch_num = 0
    """
    elif restore_model == True :
        restorer = tf.train.Saver()
        print("Restorer initialized!")
        restorer.restore(session, tf.train.latest_checkpoint(directory))
        print("Weights and biases retrieved from {} in {}\n".format(last_version, directory))
        batch_num = new_batch_num
        num_epochs = new_num_epochs
    """

    saver = tf.train.Saver()
    print("Checkpoint saver initialized!\n")
    writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(logs_path))

    print("Training model...\n")

    epoch = 0
    epoch_list = fish_filenames.copy()
    image_count = 0
    while epoch < num_epochs :
        batch_img_foveas = None
        for _ in range(5) :
            handle = epoch_list.pop(np.random.randint(0,len(epoch_list),1)[0])
            coordinates = master_dict.get(handle).get('foveas_offsets')
            img = ndimage.imread(handle, mode = 'RGB')
            image_foveas = None
            for yx_coord in coordinates :
                fovea = img[yx_coord[0]:(yx_coord[0]+fovea_presize), yx_coord[1]:yx_coord[1]+fovea_presize, :]
                fovea = misc.imresize(fovea, size = [fovea_size, fovea_size, num_channels])
                try :
                    image_foveas = np.concatenate([image_foveas, np.expand_dims(fovea, 0)], 0)
                except :
                    image_foveas = np.expand_dims(fovea, 0)
            try :
                batch_img_foveas = np.concatenate([batch_img_foveas, image_foveas], 0)
            except :
                batch_img_foveas = image_foveas
            image_count += 1
        batch_img_foveas =( batch_img_foveas / 255.0) # do not centre at mean zero.  range from 0 to 1 to match with sigmoid activation

        feed_dict = {batch_fovea : batch_img_foveas}

        if (image_count % 500 )== 0 :  #multiple of batch_range above
            _, summary, I, O = session.run([train_op, summaries,batch_fovea, decoded], feed_dict = feed_dict)
            I = I[4,:,:,:]
            O = O[4,:,:,:]
            I = (I) * 255.0
            O = (O) * 255.0
            fd.show_panel(I)
            fd.show_panel(O)
            writer.add_summary(summary, image_count)
        else :
            _, summary = session.run([train_op, summaries], feed_dict = feed_dict)
            writer.add_summary(summary, image_count)

        if len(epoch_list) < 5 :  # match the batch range above
            epoch_list = epoch_list + fish_filenames # re-stock epoch list
            epoch += 1
            print("Epoch {} finished!".format(epoch))
            saver.save(session, 'model_checkpoints/'+version_ID, global_step = image_count)

    print("\nTRAINING FINISHED!\n\nSaving final model...")
    saver.save(session, 'FINAL_MODELS/'+version_ID)
