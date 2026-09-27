from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import requests
import uuid
import json
from datetime import datetime
import asyncio
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:1b"

processing_results = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(" Starting Prodect Description API")

    yield

    logger.info("Shutting down Product Description API")

app = FastAPI(
    title="Product Description Generator API",
    description="API برای تولید عنوان، توضیح و کلمات کلیدی",
    version="1.0.0",
    lifespan=lifespan
)

class ProductRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=500, description="متن عنوان محصول")
    request_id: Optional[str] = Field(None, description="شناسه درخواست (اختیاری)")

class ProductResponse(BaseModel):
    request_id: str = Field(..., description="شناسه درخواست")
    status: str = Field(..., description="وضعیت پردازش")
    optimized_title: Optional[str] = Field(None, description="عنوان بهینه شده")
    short_description: Optional[str] = Field(None, description="توضیح کوتاه محصول")
    keywords: Optional[str] = Field(None, description="کلمات کلیدی")
    created_at: datetime = Field(..., description="زمان ایجاد")
    processing_time: Optional[float] = Field(None, description="زمان پردازش (ثانیه)")

class StatusResponse(BaseModel):
    request_id: str = Field(..., description="شناسه درخواست")
    status: str = Field(..., description="وضعیت پردازش")
    progress: float = Field(..., description="پیشرفت (۰ تا ۱)")
    estimated_time: Optional[float] = Field(None, description="زمان تخمینی باقی‌مانده (ثانیه)")

def call_ollama(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
    """Ollama تماس با مدل"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            timeout=1200
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        else:
            logger.error(f"خطا در ارتباط با مدل : {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"خطا در ارتباط با Ollama: {e}")
        return None

def extract_product_title(raw_title:str) -> Optional[str]:
    """استخراج عنوان مناسب از عنوان خام"""
    
    system_prompt = """وظیفه شما استخراج عنوان مناسب و استاندارد از عنوان‌های خام محصولات است.
    
قوانین استخراج عنوان:
۱. حذف اطلاعات اضافی مانند قیمت، سال تولید، کد محصول (مگر ضروری باشد)
۲. حفظ نام برند و ویژگی‌های اصلی محصول
۳. استفاده از ساختار استاندارد: [نام محصول] + [ویژگی اصلی] + [برند] (اختیاری)
۴. کوتاه و گویا بودن (حداکثر ۷ کلمه)
۵. استفاده از کلمات کلیدی مهم در ابتدای عنوان
۶. حذف کلمات تکراری و زائد

نمونه‌های ورودی و خروجی:

ورودی: "ماست کفی مهمان چربی کم چوب بسته بندی ۵۰۰ گرمی تولید ۱۴۰۳"
خروجی: "ماست کم چرب مهمان"

ورودی: "شامپو حجم دهنده پریشفت کلیر ضد ریزش مو بسته ۴۰۰ میلی لیتر"
خروجی: "شامپو ضد ریزش مو و حجم دهنده کلیر"

ورودی: "کفش ورزشی نایک ایرمکس مشکی سایز ۴۲ مدل ۲۰۲۴ اصل"
خروجی: "کفش ورزشی نایک ایرمکس مشکی"

ورودی: "قهوه ترک عربیکا درجه یک دم‌آوری عقلی ۲۵۰ گرم"
خروجی: "قهوه ترک عربیکا درجه یک"

ورودی: "تلموزین الچی ۵۵ اینچ اولد سامسونگ مدل ۲۰۲۳"
خروجی: "تلموزین الچی ۵۵ اینچ اولد سامسونگ"

حالا عنوان زیر را بر اساس قوانین فوق بهینه کن:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_title}
    ]
    return call_ollama(messages, temperature=0.3, max_tokens=100)

def generate_short_description(product_title: str, raw_title: str) -> Optional[str]:
    """مدل ۲: تولید توضیح کوتاه ۴۰۰ کاراکتری"""
    
    system_prompt = """یک متخصص تولید محتوای محصولات هستید. وظیفه شما نوشتن توضیح کوتاه و جذاب برای محصولات است.

الزامات توضیح کوتاه:
۱. طول دقیقاً بین ۳۵۰ تا ۴۵۰ کاراکتر (با فاصله)
۲. شروع جذاب و گیرا
۳. ذکر ۳-۴ ویژگی اصلی محصول
۴. اشاره به مزایای استفاده از محصول
۵. پایان با فرا خوان به عمل مناسب
۶. استفاده از کلمات مثبت و تشویق‌کننده
۷. نگارش روان و بدون غلط املایی
۸. مناسب برای توضیح محصول در سایت‌های فروشگاهی

نمونه‌های توضیح کوتاه:

محصول: "چای سبز بهاره لاهیجان"
توضیح: "چای سبز بهاره لاهیجان با عطر و طعم بی‌نظیر، سرشار از آنتی‌اکسیدان‌های طبیعی است. برای پیشگیری از مهمان‌های ناخوانده و حفظ سلامت بدن، انتخاب ایده‌آل است."

محصول: "موس ضد تعریق فوم‌دار نو"
توضیح: "موس ضد تعریق نو با فرمول بی‌نظیر، تا ۴۸ ساعت از تعریق و بوی نامطبوع جلوگیری می‌کند. حاوی ویتامین E و عصاره‌های گیاهی، مناسب برای پوست‌های حساس."

محصول: "کوله پشتی مدرسه طرح اسبابی‌درمن"
توضیح: "کوله پشتی مدرسه با طراحی شاد و رنگ‌های جذاب، باعث علاقه کودک به مدرسه رفتن می‌شود. اندازه مناسب برای دانش‌آموزان عزیز، دارای چندین جیب و بخش‌های مختلف است."

حالا برای محصول زیر یک توضیح کوتاه بنویس:"""

    user_prompt = f"""عنوان بهینه شده: {product_title}
عنوان اصلی: {raw_title}

لطفاً یک توضیح کوتاه جذاب بین ۳۵۰ تا ۴۵۰ کاراکتر بنویسید."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return call_ollama(messages, temperature=0.3, max_tokens=100)

def generate_keywords(product_title: str, raw_title: str, short_description: str) -> Optional[str]:
    """مدل ۳: تولید کلمات کلیدی"""
    
    system_prompt = """صص سلو و بهینه‌سازی کلمات کلیدی هستید. وظیفه شما تولید کلمات کلیدی مرتبط برای محصولات است.

قوانین تولید کلمات کلیدی:
۱. تولید ۸-۱۲ کلمه کلیدی
۲. کلمات کلیدی باید شامل موارد زیر باشد:
   - کلمه کلیدی اصلی (طولیانی)
   - کلمات کلیدی ثانویه
   - کلمات مرتبط با برند
   - کلمات توصیفی محصول
۳. کلمات باید به زبان فارسی و مرتبط با محصول باشند
۴. از کلمات عملی و بی‌ربط پرهیز شود
۵. کلمات را با کاما (،) جدا کن
۶. کلمات تکراری نباشند

فرمت خروجی:
کلمه کلیدی اصلی، کلمه کلیدی ثانویه، کلمه مرتبط ۱، کلمه مرتبط ۲ ...

نمونه‌های خروجی:

برای "کفش ورزشی نایک ایرمکس":
کفش نایک ایرمکس، کفش ورزشی مردانه، نایک اصل، کفش رانینگ، کفش پیاده‌روی، کتانی نایک، کفش اسپرت

برای "قهوه ترک عربیکا":
قهوه ترک اصل، قهوه عربیکا، قهوه دم‌آوری، قهوه درجه یک، قهوه لذیذ، قهوه ایرانی، قهوه خانگی

برای "شامپو کلیر ضد ریزش":
شامپو ضد ریزش مو، شامپو کلیر، شامپو حجم دهنده، تقویت مو، شامپو پریشفت، مراقبت مو، شامپو ایرانی

برای "ماست کم چرب مهمان":
ماست مهمان، ماست کم چرب، ماست، ماست پروبیوتیک، لبنیات مهمان، ماست مکع، ماست خوشمزه

حالا برای محصول زیر کلمات کلیدی تولید کن:"""

    user_prompt = f"""عنوان محصول: {product_title}
عنوان اصلی: {raw_title}
توضیح کوتاه: {short_description}

کلمات کلیدی مناسب برای این محصول را تولید کن."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return call_ollama(messages, temperature=0.5, max_tokens=150) 

async def process_product_description(request_id: str, product_text: str):
    """پردازش فرم‌مزمان برای تولید محتوای محصول"""
    global processing_results
    start_time = datetime.now()

    try:
        logger.info(f"[{request_id}] مرحله ۱: استخراج عنوان...")
        processing_results[request_id]["progress"] = 0.2

        optimized_title = extract_product_title(product_text)
        if not optimized_title:
            raise Exception("خطا در استخراج عنوان")

        processing_results[request_id]["optimized_title"] = optimized_title
        processing_results[request_id]["progress"] = 0.4

        logger.info(f"[{request_id}] مرحله ۲: تولید توضیح کوتاه...")
        processing_results[request_id]["progress"] = 0.5

        short_description = generate_short_description(optimized_title, product_text)
        if not short_description:
            raise Exception("خطا در تولید توضیح کوتاه")

        processing_results[request_id]["short_description"] = short_description
        processing_results[request_id]["progress"] = 0.7

        logger.info(f"[{request_id}] مرحله ۳: تولید کلمات کلیدی...")
        processing_results[request_id]["progress"] = 0.8

        keywords = generate_keywords(optimized_title, product_text, short_description)
        if not keywords:
            raise Exception("خطا در تولید کلمات کلیدی")

        processing_results[request_id]["keywords"] = keywords
        processing_results[request_id]["progress"] = 1.0

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        processing_results[request_id]["processing_time"] = processing_time
        processing_results[request_id]["status"] = "completed"

        logger.info(f"[{request_id}] پردازش کامل شد در {processing_time:.2f} ثانیه")

    except Exception as e:
        logger.error(f"[{request_id}] خطا در پردازش: {e}")
        processing_results[request_id]["status"] = "failed"
        processing_results[request_id]["error"] = str(e)

@app.get("/")
async def root():
    """API صفحه اصلی"""
    return {
        "message": "Product Description Generator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "ایجاد محتوای جدید برای محصول",
            "GET /status/{request_id}": "بررسی وضعیت پردازش",
            "GET /result/{request_id}": "دریافت نتایج پردازش"
        },
        "docs": "/docs"
    }

@app.post("/generate", response_model=ProductResponse, status_code=202)
async def generate_description(product_request: ProductRequest, background_tasks: BackgroundTasks):
    """
    ایجاد درخواست جدید برای تولید محتوای محصول
    
    دریافت متن محصول و شروع پردازش غیرهمزمان
    """
    global processing_results
    request_id = product_request.request_id or str(uuid.uuid4())

    processing_results[request_id] = {
        "status": "processing",
        "progress": 0.0,
        "created_at": datetime.now(),
        "product_text": product_request.text,
        "optimized_title": None,
        "short_description": None,
        "keywords": None,
        "processing_time": None
    }

    background_tasks.add_task(process_product_description, request_id, product_request.text)

    logger.info(f"درخواست جدید ایجاد شد: {request_id}")

    return ProductResponse(
        request_id=request_id,
        status="processing",
        created_at=processing_results[request_id]["created_at"],
        processing_time=None
    )

@app.get("/status/{request_id}", response_model=StatusResponse)
async def get_status(request_id: str):
    """
    بررسی وضعیت پردازش یک درخواست
    
    نمایش پیشرفت پردازش
    """
    global processing_results
    print(processing_results)

    if request_id not in processing_results:
        raise HTTPException(status_code=404, detail="شناسه درخواست یافت نشد")

    result = processing_results[request_id]

    estimated_time = None
    if result["status"] == "processing":
        elapsed = (datetime.now() - result["created_at"]).total_seconds()
        if result["progress"] > 0:
            estimated_time = (elapsed / result["progress"]) * (1 - result["progress"])

    return StatusResponse(
        request_id=request_id,
        status=result["status"],
        progress=result["progress"],
        estimated_time=estimated_time
    )

@app.get("/result/{request_id}", response_model=ProductResponse)
async def get_result(request_id: str):
    """
    دریافت نتایج پردازش
    
    در صورت تکمیل پردازش، نتایج کامل را برمی‌گرداند
    """
    global processing_results
    if request_id not in processing_results:
        raise HTTPException(status_code=404, detail="شناسه درخواست یافت نشد")

    result = processing_results[request_id]

    if result["status"] == "processing":
        raise HTTPException(status_code=202, detail="پردازش هنوز کامل نشده است")

    if result["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"خطا در پردازش: {result.get('error', 'خطای ناشناخته')}"
        )

    return ProductResponse(
        request_id=request_id,
        status=result["status"],
        optimized_title=result["optimized_title"],
        short_description=result["short_description"],
        keywords=result["keywords"],
        created_at=result["created_at"],
        processing_time=result["processing_time"]
    )

@app.get("/health")
async def health_check():
    """بررسی سلامت API و اتصال به Ollama"""

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        if response.status_code == 200:
            return {
                "status": "healthy",
                "ollama": "connected",
                "message": "API و Ollama سالم هستند"
            }

        return {
            "status": "unhealthy",
            "ollama": "error",
            "message": f"Ollama status code: {response.status_code}"
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "unhealthy",
            "ollama": "disconnected",
            "message": "اتصال به Ollama برقرار نشد"
        }

    except requests.exceptions.Timeout:
        return {
            "status": "unhealthy",
            "ollama": "timeout",
            "message": "زمان اتصال به Ollama تمام شد"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama": "error",
            "message": str(e)
        }