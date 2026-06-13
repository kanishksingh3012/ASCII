import reflex as rx

config = rx.Config(
    app_name="ascii_studio",
    # Larger upload limit for video; mirrors the old Streamlit cap intent.
    # (Reflex serves uploads through its own backend route.)
)
