
import asyncio
import edge_tts
import pygame
import time

async def tao_giong_noi():
    giong = edge_tts.Communicate(
        text="Xin chào, tôi có thể nhận diện người và ô tô ở phía trước.",
        voice="vi-VN-HoaiMyNeural"
    )
    await giong.save("giong_viet.mp3")

try:
    asyncio.run(tao_giong_noi())
    print("Đã tạo file giọng nói.")

    pygame.mixer.init()
    pygame.mixer.music.load("giong_viet.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()
    print("Đọc giọng nói thành công.")

except Exception as loi:
    print("Lỗi:", loi)
