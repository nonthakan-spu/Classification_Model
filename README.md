# มินิโปรเจคการสร้าง Classification Model เพื่อแยกแยะโรคในทุเรียนจากใบ
## จุดประสงค์
เพื่อทำการแยกแยะโรคที่เกิดขึ้นในทุเรียนจากใบของทุเรียนด้วยโมเดล CNN โดยการทำ Fine-tuned โมเดล EfficientNet-B0 
## ขั้นตอนการทำงาน
### 1.import ไลบรารี่ที่จำเป็น และกำหนดค่าพื้นฐาน
ทำการ import ไลบรารี่ที่จำเป็นต้องใช้ และตั้งค่าตัวแปรต่างๆ
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# กำหนดค่าพารามิเตอร์ต่างๆ
IMG_SIZE = 224  # ขนาดรูปภาพที่ EfficientNetB0 ต้องการ
BATCH_SIZE = 32 # จำนวนข้อมูลที่ส่งให้โมเดลในแต่ละรอบ
EPOCHS = 50      # จำนวนรอบในการเทรนโมเดล
```
### 2.เตรียมชุดข้อมูล
นำชุดข้อมูลที่มีมาทำการแบ่งออกเป็น 3 ชุดด้วยกันคือ train, validation และ test ซึ่งโดยทั่วไปแล้วจะถูกแบ่งเป็น traint 70%, validation 10% และ test 20% ถ้าในกรณีที่ชุดข้อมูลมีน้อยหรือจำกัดมากสามารถทำการ Augmented เพื่อเพิ่มข้อมูลที่ใช้ในการฝึกได้ ในที่นี้จะทำการ augmentation เพื่อเพิ่มข้อมูลด้วย
```python
# สร้างเลเยอร์สำหรับเพิ่มข้อมูล augmented
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# โหลดข้อมูลจาก train, test_1, และ test_2 แยกกันก่อน
train_full_ds_raw = tf.keras.utils.image_dataset_from_directory(
    directory = DATASET_DIR,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = None, # ไม่ต้องใส่ batch_size ตอนโหลดครั้งแรก
    shuffle=False, # ไม่ต้อง shuffle ตอนโหลดครั้งแรก
    seed = 42
)

test_1_full_ds_raw = tf.keras.utils.image_dataset_from_directory(
    directory = DATASET_TEST_1,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = None,
    shuffle=False,
    seed = 42
)

test_2_full_ds_raw = tf.keras.utils.image_dataset_from_directory(
    directory = DATASET_TEST_2,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = None,
    shuffle=False,
    seed = 42
)

# รวม Datasets ทั้งหมดเข้าด้วยกัน
all_ds_raw = train_full_ds_raw.concatenate(test_1_full_ds_raw).concatenate(test_2_full_ds_raw)

# นับจำนวนข้อมูลทั้งหมดที่มี
total_files = tf.data.experimental.cardinality(all_ds_raw).numpy()
print(f"จำนวนไฟล์รูปภาพทั้งหมด: {total_files}")

# คำนวณจำนวนข้อมูลสำหรับแต่ละชุดใหม่
train_size = int(TRAIN_SPLIT * total_files)
val_size = int(VAL_SPLIT * total_files)
test_size = total_files - train_size - val_size

print(f"จำนวนข้อมูล Train ใหม่: {train_size}")
print(f"จำนวนข้อมูล Validation ใหม่: {val_size}")
print(f"จำนวนข้อมูล Test ใหม่: {test_size}")

# ทำการสุ่มข้อมูลก่อนแบ่ง
all_ds_raw = all_ds_raw.shuffle(buffer_size=total_files, seed=42)

# แบ่งข้อมูลโดยใช้ take และ skip
train_ds = all_ds_raw.take(train_size)
remaining_ds = all_ds_raw.skip(train_size)

val_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size) 

# ตรวจสอบจำนวนข้อมูลในแต่ละชุดหลังจากแบ่ง
print(f"\nจำนวนข้อมูลใน train_ds: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"จำนวนข้อมูลใน val_ds: {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"จำนวนข้อมูลใน test_ds: {tf.data.experimental.cardinality(test_ds).numpy()}")
```
### 3.การสร้าง Pipeline ของชุดข้อมูล
การสร้าง Pipeline ของชุดข้อมูลนั้นเพื่อความง่ายและสะดวกในการจัดการข้อมูล เพื่อไม่ใช้เกิดข้อมูลผิดพลาดก่อนนำไปฝึกโมเดล
```python
# สร้างตัวแปรเก็บจำนวนคลาสทั้งหมด
NUM_CLASSES = len(train_full_ds_raw.class_names)

# ฟังก์ชันสำหรับปรับขนาดและประมวลผลรูปภาพให้พร้อมสำหรับโมเดล
def preprocess_image(image, label):
    image = tf.keras.applications.efficientnet.preprocess_input(image) # ฟังก์ชันเฉพาะของ EfficientNet
    return image, label

# สร้าง Data Pipeline
# สำหรับ Train: เพิ่มข้อมูล (augment) -> ปรับขนาด (preprocess) -> เตรียมข้อมูลล่วงหน้า (prefetch) -> Batch
train_ds_aug = (train_ds
            .cache()
            .shuffle(buffer_size=1000)
            .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE) # เพิ่ม Batching หลัง Augmentation
            .prefetch(tf.data.AUTOTUNE))

# สำหรับ Validation และ Test: ปรับขนาด (preprocess) -> เตรียมข้อมูลล่วงหน้า (prefetch) -> Batch
val_ds  = (val_ds
           .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE) # เพิ่ม Batching
           .cache().prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds
           .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE) # เพิ่ม Batching
           .cache().prefetch(tf.data.AUTOTUNE))
```
### 4.โหลดโมเดล EfficientNet-B0 และ Freezing Layers
ทำการโหลดโมเดล EfficientNet-B0 โดยไม่ต้องโหลดส่วนของ Top Layers มาด้วย เนื่องจากเราจะสร้างในส่วนของ Top Layers เองเพื่อให้จำนวนคลาส Output เป็นไปตามจำนวนคลาสของชุดข้อมูลของเรา และทำการ Freezing Layers เพื่อหยุดการเรียนรู้ของ Layers ล่างทั้งหมดเพื่อไม่ให้ได้รับผลกระทบจากชุดข้อมูลใหม่ที่เรากำลังจะให้โมเดลฝึก เนื่องจากเราจำเป็นต้องใช้ความสามารถของ Layer ล่างๆ ในการแยกแยะขอบ มุม หรือ สีของใบที่โมเดลได้ถูกฝึกมาด้วยชุดข้อมูลอันมหาศาล
```python
# โหลด EfficientNetB0 ที่เทรนด้วย 'imagenet'
# include_top=False หมายถึง ไม่เอา Classification Head ส่วนบนสุดมาด้วย
base_model_aug = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# ทำการ "แช่แข็ง" ไม่ให้เลเยอร์ของ base_model_aug ถูกเทรนใหม่
base_model_aug.trainable = False
```
### 5.สร้าง Output Layers
สร้างส่วนของ Top Layers เพื่อใช้ในการฝึกชุดข้อมูลใหม่
```python
# สร้างโมเดลใหม่โดยเริ่มจาก base_model_aug ที่เราโหลดมา
inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model_aug(inputs, training=False) # training=False เพื่อให้ BatchNormalization ทำงานใน inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x) # ลดมิติของ feature map
x = tf.keras.layers.Dropout(0.2)(x) # ป้องกัน Overfitting
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model_aug = tf.keras.Model(inputs, outputs)
```
### 6.ทำการคอมไฟล์โมเดลและฝึกโมเดล
สร้างส่วนของ Callbacks ขึ้นมาเพื่อสร้าง checkpoint, early stop และ ปรับอัตราการเรียนรู้เมื่อผ่านไปสักระยะ ทำการคอมไฟล์โมเดลและนำชุดข้อมูล train มาทำการฝึกโมเดลและชุดข้อมูล validation เพื่อทดสอบระหว่างเรียนรู้
```python
# กำหนด Optimizer, Loss Function, และ Metrics
model_aug.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy', # เหมาะสำหรับโจทย์ 2 คลาส
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# แสดงสรุปโครงสร้างของโมเดลทั้งหมด
model_aug.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_aug.keras", save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7, monitor="val_loss"),
]

print("\n--- เริ่มการเทรนโมเดล ---")
history_aug = model_aug.fit(
    train_ds_aug,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)
print("--- การเทรนโมเดลเสร็จสิ้น ---\n")
```
### 7.การวัดผล
นำชุดข้อมูล test มาทำการทดสอบโมเดลเพื่อวัดผลโมเดลว่าการเรียนรู้ของชุดข้อมูล train นั้นสามารถนำมาใช้งานบนชุดข้อมูลที่ไม่เคยเห็นได้ดีแค่ไหน
```python
print("--- เริ่มการวัดผลโมเดลด้วย Test Set ---")
# รับค่าทั้งหมด 3 ค่าตามที่กำหนดใน metrics
loss, accuracy, auc = model_aug.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")
```
## สรุปผลการทดลอง
จากการทดลองพบว่าเมื่อทำการ augmentation ข้อมูลก่อนนำไปเทรนจะทำให้ระยะเวลาในการเทรนเพิ่มขึ้นอย่างเห็นได้ชัด แต่จำนวนรอบในการเทรนนั้นน้อยกว่าโมเดลที่ถูกเทรนด้วยข้อมูลที่ไม่ผ่านการ augmentation ส่วนความแม่นยำนั้นไม่ต่างกันมาก อยู่ที่ราวๆ 99.7% - 99.9% ทั้งโมเดลที่เทรนด้วยข้อมูล 2,818 ภาพ และโมเดลที่ถูกเทรนด้วยข้อมูล 1,000 ภาพ 
