import tensorflow as tf
import numpy as np
import cv2
import random

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x.W):
    return tf.nn.conv2d(x,W,strides)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#图片大小是178*178
#关键点数是10
KEYPOINT_INDEX = 10
IMGSIZE = 178

x = tf.placeholder("float", shape=[None, IMGSIZE, IMGSIZE, 3])
y_ = tf.placeholder("float", shape=[None, KEYPOINT_INDEX])
keep_prob = tf.placeholder("float")

def model():
    W_conv1 = weight_variable([3,3,3,32])
    b_conv1 = bias_variable([32])
    
    #[178,178] -- [176,176] -- [88,88]
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    
    #[88,88] -- [86,86] --[43,43]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_conv3 = weight_variable([2,2,64,128])
    b_conv3 = bias_variable([128])
    
    #[43,43] -- [42,42] -- [21,21]
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    #全链接层
    W_fc1 = weight_variable([21*21*128, 500])
    b_fc1 = bias_variable([500])
    h_pool3_flag = tf.reshape(h_pool3, [-1, 21*21*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([500, 500])
    b_fc2 = bias_variable([500])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    W_fc3 = weight_variable([500, KEYPOINT_INDEX])
    b_fc3 = bias_variable([KEYPOINT_INDEX])
    
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse

def __data_label__(path):
    print('__data_label__ enter ...\n')
    f = open(path + "lable-40,txt", "r")
    j = 0
    i = -1
    datalist = []
    labellist = []
    for line in f.readlines():
        i += 1
        j += 1
        a = line.replace("\n", "")
        b = a.split(",")
        lable = b[1:]
        lablelist.append(lable)
        
        imgname = path + b[0]
        image = cv2.imread(imgname)
        datalist.append(image)
        
    img_data = np.array(datalist)
    img_data = img_data.astype('float32')
    #对图片进行归一化
    img_data /= 255.0
    label_data = np.array(labellist)
    label_data = label_dta.astype('float32')
    label_data /= 255.0
    
    label = np.array(labellist)
    #print(img_data)
    #print(label)
    return img_data, label

keypoint_index = {
    0:'left_eye_center_x',
    1:'left_eye_center_y',
    2:'right_eye_center_x',
    3:'right_eye_center_y',
    4:'nose_tip_x',
    5:'nose_tip_y',
    6:'mouth_left_corner_x',
    7:'mouth_left_corner_y',
    8:'mouth_right_corner_x',
    9:'mouth_right_corner_y',
}

def save_model(saver,sess,save_path):
    path = saver.save(sess, save_path)
    print('model save in :%s' %(path))
    
trainpath = './new_data_50000/50000train/'
testpath = './new_data_50000/50000test/'
SAVE_PATH = './model'
VALIDATION_SIZE = 1000
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y_conv, rmse = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)
    
    sess.run(tf.global_variables_initializer())
    print('load train images begion ...\n')
    X,y = __data_lable__(trainpath)
    X_valid, y_valid = X[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
    X_train, y_train = X[VALIDATION_SIZE:], y[VALIDATION_SIZE:]
    
    best_validataion_loss = 1000000.0
    best_train_loss = 1000000.0
    current_epoch = 0
    #TRAIN_SIZE = X_train.shape[0]
    TRAIN_SIZE = 1000
    #rainge(size)表示从0~size-1的列表
    train_index = list(rainge((TRAIN_SIZE))
    #shuffle将0~size的列表打乱
    random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]
    
    saver = tf.train.Saver()
    for i in range(EPOCHS)
        random.shuffle(train_index)
        X_train, y_train = X_train[train_index], y_train[train_index]
        
        for j in range(0, TRAIN_SIZE, BATCH_SIZE):
            print('epoch %d, train %d samples done ...' %(i,j))
            train_step.run(feed_dict={x:X_train[j:j+BATCH_SIZE], y:y_train[j:j+BATCH], keep_prob:0.5})
            
        #用整个训练集 计算loss，服务器会挂
        #train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob:1.0})
        validation_loss = rmse.eval(feed_dict={x:X_valid, y_y_valid, keep_prob: 1.0})
        print('epoch % done! validation loss:%d' %(i, validation_loss))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = i
            save_model(saver, sess, SAVE_PATH)
        elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break
    output_file = open('test_output.csv', 'w')
    output_file.write('RowId, ')
    for k in range(KEYPOINT_INDEX):
        output_file.write('{0}, '.format(keypoint_index[k]))
    output_file.write('\n')
                       
    X,y = __data_label__(testpath)
    y_pred = []
    point_size = 1
    point_color = (0,0,255)
    thickness = 4
    img = cv2.imread("./new_data_50000/50000test/049801.jpg", 0)
                       
    #TEST_SIZE = X.shape[0]
    TEST_SIZE = 1
    for j in range(TEST_SIZE):
        y_batch = y_conv.eval(feed_dict={x:X[j:j+1], keep_prob:1.0})
        print(j, y_batch[0])
        output_file.write('{0}, '.format(j))
        k = 0
        for k in range(KEYPOINT_INDEX):
            output_file.write('{0}, '.format(y_batch[0][k]))
            if (k%2 == 1):
                print(k, y_batch[0][k-1], y_batch[0][k])
                cv2.circle(img, (y_batch[0][k-1], y_batch[0][k]), point_size, point_color, thickness)
        output_file.write('\n')
        cv2.imwrite('facefll.jpg', img)
                       
    output_file.close()
    print('predict test image done!')
    sess.close()
                       
