import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { prompt } = await request.json();
    
    // Simulate AI generation time
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    return NextResponse.json({ 
      success: true, 
      message: `Generated CAD for: ${prompt}`,
      csgTree: [
        { id: '1', type: 'box', params: { x: 20, y: 20, z: 20 } }
      ]
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to generate CAD model' }, { status: 500 });
  }
}
