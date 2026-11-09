from fastapi import APIRouter, Body, HTTPException

from .. import catalog

router = APIRouter()


@router.get('/api/catalog')
def get_catalog():
    return catalog.full_catalog()


@router.get('/api/model-classes')
def model_classes():
    return catalog.model_classes_info()


@router.get('/api/item/{item_id}')
def get_item(item_id: str):
    found = catalog.get_item(item_id)
    if not found:
        raise HTTPException(404, 'Item not found')
    return found


@router.post('/api/ask')
async def ask(payload: dict = Body(...)):
    return catalog.ask(payload)
