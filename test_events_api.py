"""
Test script for Event API endpoints
Run this after starting the server
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_list_events():
    """Test GET /api/events endpoint"""
    print("\n🔍 Testing: GET /api/events")
    print("-" * 60)
    
    response = requests.get(f"{BASE_URL}/api/events")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        events = response.json()
        print(f"Total Events: {len(events)}")
        print("\nEvents:")
        for event in events:
            print(f"  - ID: {event['event_id']}")
            print(f"    Classes: {', '.join(event.get('detected_classes', []))}")
            print(f"    Duration: {event.get('duration_seconds', 'N/A')}s")
            print(f"    Has Video: {event.get('has_video', False)}")
            print()
        return events
    else:
        print(f"Error: {response.text}")
        return []


def test_get_metadata(event_id):
    """Test GET /api/events/{event_id}/metadata endpoint"""
    print(f"\n📋 Testing: GET /api/events/{event_id}/metadata")
    print("-" * 60)
    
    response = requests.get(f"{BASE_URL}/api/events/{event_id}/metadata")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        metadata = response.json()
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))
        return metadata
    else:
        print(f"Error: {response.text}")
        return None


def test_stream_video(event_id):
    """Test GET /api/events/{event_id}/video endpoint"""
    print(f"\n🎥 Testing: GET /api/events/{event_id}/video")
    print("-" * 60)
    
    response = requests.get(f"{BASE_URL}/api/events/{event_id}/video", stream=True)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    if response.status_code == 200:
        # Get content length
        content_length = response.headers.get('content-length')
        if content_length:
            print(f"Video Size: {int(content_length) / 1024:.2f} KB")
        else:
            # If no content-length, read the stream
            total_bytes = sum(len(chunk) for chunk in response.iter_content(8192))
            print(f"Video Size: {total_bytes / 1024:.2f} KB")
        print("✅ Video stream successful")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    print("\n" + "=" * 60)
    print("🧪 Event API Endpoint Tests")
    print("=" * 60)
    
    try:
        # Test 1: List all events
        events = test_list_events()
        
        if events:
            # Test 2: Get metadata for first event
            first_event_id = events[0]['event_id']
            test_get_metadata(first_event_id)
            
            # Test 3: Stream video for first event
            if events[0].get('has_video'):
                test_stream_video(first_event_id)
            else:
                print(f"\n⚠️  No video available for event {first_event_id}")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server. Is it running?")
        print("Start the server with: python main.py")
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print()


if __name__ == "__main__":
    main()
