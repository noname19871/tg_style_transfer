"""
This is a echo bot.
It echoes any incoming text messages.
"""

import os
from rq import Queue
from rq.job import Job

from style_transfer.model import VGGStyleTransfer
from redis_worker import conn

import aiogram.utils.markdown as md

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ParseMode
from aiogram.utils import executor

API_TOKEN = os.getenv('BOT_API_TOKEN')

bot = Bot(token=API_TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

redis_queue = Queue(connection=conn, default_timeout=600)
style_transfer_model = VGGStyleTransfer()


# States
class Form(StatesGroup):
    content_photo = State()
    style_photo = State()
    job_id = State()


@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await message.reply("Hi there! I can change your photos style! Use /transfer command and let's go!")


@dp.message_handler(commands='transfer')
async def transfer_start(message: types.Message):
    await Form.content_photo.set()

    await bot.send_message(message.chat.id, "Send me content photo")


# You can use state '*' if you need to handle all states
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return

    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Cancelled. You can start again using /transfer command',
                        reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=Form.content_photo, content_types=['photo'])
async def process_content_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['content_photo'] = message.photo[-1]

    await Form.next()
    await bot.send_message(message.chat.id, "Send me style photo")


@dp.message_handler(state=Form.content_photo)
async def wrong_content_photo(message: types.Message):
    await message.reply("I need a content photo to work. Send me content photo")


@dp.message_handler(state=Form.style_photo, content_types=['photo'])
async def process_content_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['style_photo'] = message.photo[-1]
        await data['content_photo'].download(f'./photos/{message.chat.id}_content.jpg')
        await data['style_photo'].download(f'./photos/{message.chat.id}_style.jpg')

        content_photo = style_transfer_model.load_image(f'./photos/{message.chat.id}_content.jpg')
        style_photo = style_transfer_model.load_image(f'./photos/{message.chat.id}_style.jpg')
        result_path = f'./photos/{message.chat.id}_result.jpg'
        job = redis_queue.enqueue_call(
            func=style_transfer_model.run_style_transfer,
            args=(content_photo, style_photo, result_path),
            result_ttl=600,
            ttl=600,
            failure_ttl=600
        )
        data['job_id'] = job.get_id()
        await Form.next()

    await bot.send_message(
        message.chat.id,
        md.text(
            md.text("You're all set!"),
            md.text("Now I will start to prepare your photo."),
            sep='\n',
        ),
        parse_mode=ParseMode.MARKDOWN,
    )


@dp.message_handler(state=Form.style_photo)
async def wrong_style_photo(message: types.Message):
    await message.reply("I need a style photo to work. Send me content photo")


@dp.message_handler(commands='check', state=Form.job_id)
async def transfer_start(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        job = Job.fetch(data['job_id'], connection=conn)
        if job.is_finished:
            resulting_image = open(f'./photos/{message.chat.id}_result.jpg', 'rb')
            await bot.send_photo(message.chat.id, resulting_image, "This is your photo")
            await state.finish()
        else:
            await bot.send_message(message.chat.id, "Your photo isn't ready. Keep waiting :)")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
